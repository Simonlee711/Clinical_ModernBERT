#!/usr/bin/env python
import os
import re
import glob
import gzip
os.environ["HF_HOME"] = "/data2/simon/BioClinical_ModernBERT/src"
os.environ["HF_HUB_CACHE"] = "/data2/simon/BioClinical_ModernBERT/src/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import random
import pandas as pd
import numpy as np
import torch
import swifter
from tqdm import tqdm
from lxml import etree
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from typing import Optional
import wandb
import math
wandb.login(key="0d3cef273ac07263f8b9035513b8693a26308dce")  # <-- Your wandb key

from concurrent.futures import ProcessPoolExecutor

from transformers import BertConfig, BertForMaskedLM
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Define the tokenizer globally so that it is available in multiprocess workers.
base_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_csv_and_clean(path, label):
    logging.info(f"Loading {label} notes from {path}")
    df = pd.read_csv(path)
    columns_to_drop = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    logging.info(f"Cleaning {label} text")
    df["clinical_text"] = df["text"].swifter.apply(clean_text)
    df = df.drop(columns="text")
    logging.info(f"{label} loaded with shape {df.shape}")
    return df

def extract_article_data(article):
    def xpath_text(elem, path):
        found = elem.find(path)
        return found.text.strip() if found is not None and found.text else None

    pmid = xpath_text(article, ".//PMID")
    title = xpath_text(article, ".//ArticleTitle")
    abstract_parts = article.findall(".//Abstract/AbstractText")
    abstract = " ".join([a.text.strip() for a in abstract_parts if a is not None and a.text]) if abstract_parts else None

    if title and abstract:
        clinical_text = f"{title} {abstract}"
    elif title:
        clinical_text = title
    elif abstract:
        clinical_text = abstract
    else:
        clinical_text = None

    return {"clinical_text": clinical_text}

def parse_pubmed_xml_file(file):
    logging.info(f"Parsing {file}")
    records = []
    with gzip.open(file, 'rb') as f:
        context = etree.iterparse(f, tag='PubmedArticle')
        for _, elem in context:
            record = extract_article_data(elem)
            if record["clinical_text"]:
                records.append(record)
            elem.clear()
    return records

def load_pubmed_and_clean(pubmed_glob, num_workers=8):
    files = glob.glob(pubmed_glob)
    logging.info(f"Found {len(files)} PubMed XML files to parse.")
    records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(parse_pubmed_xml_file, file) for file in files]
        for future in tqdm(futures, desc="Processing PubMed files"):
            records.extend(future.result())
    df = pd.DataFrame(records)
    logging.info(f"Loaded {df.shape[0]} PubMed articles with clinical text.")
    return df

def load_data():
    # Example for using PubMed data.
    pubmed_path_glob = "../../data/pubmed/pubmed25n0001.xml.gz"
    pubmed = load_pubmed_and_clean(pubmed_path_glob)
    pubmed = pubmed[["clinical_text"]]
    return pubmed

class LowPrecisionLayerNorm(nn.LayerNorm):
    def forward(self, input):
        orig_dtype = input.dtype
        if input.dtype == torch.float32:
            input = input.to(torch.bfloat16)
        out = super().forward(input)
        return out.to(orig_dtype)

class MosaicBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dropout_prob = 0.0
        self.attention_probs_dropout_prob = 0.0
        self.flash_attention = True
        self.alibi = True
        self.gated_linear_units = True
        self.use_low_precision_layernorm = True
        self.rope_theta = kwargs.get("rope_theta", 10000)
        self.context_length = kwargs.get("context_length", 1024)

class MosaicBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config: MosaicBertConfig):
        super().__init__(config)
        if config.use_low_precision_layernorm:
            for name, module in self.named_modules():
                if isinstance(module, nn.LayerNorm):
                    new_ln = LowPrecisionLayerNorm(module.normalized_shape, module.eps, module.elementwise_affine)
                    with torch.no_grad():
                        new_ln.weight.copy_(module.weight)
                        if module.bias is not None:
                            new_ln.bias.copy_(module.bias)
                    parent = self
                    *path, last = name.split('.')
                    for p in path:
                        parent = getattr(parent, p)
                    setattr(parent, last, new_ln)
    def forward(self, *args, **kwargs):
        if 'num_items_in_batch' in kwargs:
            kwargs.pop('num_items_in_batch')
        return super().forward(*args, **kwargs)

# Implement StableAdamW optimizer as per the requirements
class StableAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, clipping_threshold=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, clipping_threshold=clipping_threshold)
        super(StableAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                clipping_threshold = group['clipping_threshold']
                lr = group['lr']
                weight_decay = group['weight_decay']
                eps = group['eps']

                state['step'] += 1

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / bias_correction1
                update_norm = update.norm()
                denom_norm = denom.norm()

                if update_norm > clipping_threshold * denom_norm:
                    update.mul_(clipping_threshold * denom_norm / update_norm)

                p.data.addcdiv_(update, denom, value=-lr / math.sqrt(bias_correction2))
        return loss

def create_extended_context_dataset(dataset, max_length=8192):
    def combine_examples(examples, target_length):
        combined_texts = []
        current_text = ""
        for text in examples:
            if len(current_text) + len(text) <= target_length:
                current_text += text + " "
            else:
                if current_text:
                    combined_texts.append(current_text.strip())
                current_text = text + " "
        if current_text:
            combined_texts.append(current_text.strip())
        return combined_texts

    all_texts = dataset["clinical_text"]
    long_texts = combine_examples(all_texts, max_length)
    return Dataset.from_dict({"clinical_text": long_texts})

def upsample_quality_sources(dataset, quality_indices, upsample_factor=2.0):
    upsampled_data = []
    for idx, example in enumerate(dataset):
        upsampled_data.append(example)
        if idx in quality_indices:
            for _ in range(int(upsample_factor - 1)):
                upsampled_data.append(example)
    return Dataset.from_list(upsampled_data)

def main():
    df = load_data()
    df = df.sample(frac=0.01, random_state=42)
    logging.info(f"Using a subset of data: {df.shape[0]} records (~1% of full dataset)")
    
    avg_len = df["clinical_text"].swifter.apply(lambda x: len(x.split())).mean()
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"Average token count: {avg_len:.2f}")
    logging.info(f"DataFrame memory usage: {mem_mb:.2f} MB")
    
    df = df.reset_index(drop=True)
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return base_tokenizer(examples["clinical_text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=10, remove_columns=["clinical_text"])
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset.column_names if col.startswith("__")]
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    vocab_size_before = len(base_tokenizer)
    padding_needed = (64 - (vocab_size_before % 64)) % 64
    if padding_needed != 0:
        dummy_tokens = [f"<dummy_extra_token_{i}>" for i in range(padding_needed)]
        base_tokenizer.add_tokens(dummy_tokens, special_tokens=False)
    logging.info(f"Tokenizer vocab size is now {len(base_tokenizer)} (was {vocab_size_before})")
    
    config = MosaicBertConfig.from_pretrained("answerdotai/ModernBERT-base")
    config.vocab_size = len(base_tokenizer)
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    config.flash_attention = True
    config.alibi = True
    config.gated_linear_units = True
    config.use_low_precision_layernorm = True
    config.rope_theta = 10000
    config.context_length = 1024
    
    model = MosaicBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", config=config)
    model.resize_token_embeddings(len(base_tokenizer))
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        mlm=True,
        mlm_probability=0.30
    )
    
    wandb.init(project="mosaicbert-pretrain")
    wandb.config.update({
        "description": "MosaicBERT pretraining with 30% MLM, BF16 LN, FlashAttention, ALiBi, GLUs, dropout=0, StableAdamW optimizer"
    })
    
    initial_batch_size = 64
    
    training_args = TrainingArguments(
        output_dir="checkpoints_mosaic_bert",
        overwrite_output_dir=True,
        run_name="mosaicbert_pretrain",
        num_train_epochs=200,
        per_device_train_batch_size=initial_batch_size,
        save_strategy="steps",
        save_steps=500000,
        logging_steps=1000,
        learning_rate=8e-4,
        bf16=True,
        report_to=["wandb"],
        save_total_limit=None,
        remove_unused_columns=False,
        local_rank=-1,
        dataloader_num_workers=4,
        fp16_backend="auto",
        no_cuda=False,
        optim="adamw_torch"
    )
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = StableAdamW(
        optimizer_grouped_parameters,
        lr=8e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        clipping_threshold=1.0
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=base_tokenizer,
        optimizers=(optimizer, None)
    )
    
    trainer.train()
    
    logging.info("Starting context length extension phase")
    config.rope_theta = 160000
    config.context_length = 8192
    
    extended_dataset = create_extended_context_dataset(dataset, max_length=8192)
    
    def tokenize_extended_function(examples):
        return base_tokenizer(examples["clinical_text"], truncation=True, padding="max_length", max_length=8192)
    
    tokenized_extended_dataset = extended_dataset.map(
        tokenize_extended_function, 
        batched=True, 
        num_proc=10, 
        remove_columns=["clinical_text"]
    )
    tokenized_extended_dataset = tokenized_extended_dataset.with_format("torch")
    
    training_args_extension = TrainingArguments(
        output_dir="checkpoints_mosaic_bert_extended",
        overwrite_output_dir=True,
        run_name="mosaicbert_context_extension",
        num_train_epochs=20,
        per_device_train_batch_size=8,
        save_strategy="steps",
        save_steps=100000,
        logging_steps=1000,
        learning_rate=3e-4,
        bf16=True,
        report_to=["wandb"],
        save_total_limit=None,
        remove_unused_columns=False,
        local_rank=-1,
        dataloader_num_workers=4,
        fp16_backend="auto",
        no_cuda=False,
    )
    
    optimizer_extension = StableAdamW(
        optimizer_grouped_parameters,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        clipping_threshold=1.0
    )
    
    trainer_extension = Trainer(
        model=model,
        args=training_args_extension,
        data_collator=data_collator,
        train_dataset=tokenized_extended_dataset,
        tokenizer=base_tokenizer,
        optimizers=(optimizer_extension, None)
    )
    
    trainer_extension.train()
    
    logging.info("Starting upsampled quality sources phase")
    quality_indices = list(range(0, len(tokenized_extended_dataset) // 5))
    upsampled_dataset = upsample_quality_sources(tokenized_extended_dataset, quality_indices)
    
    trainer_final = Trainer(
        model=model,
        args=training_args_extension,
        data_collator=data_collator,
        train_dataset=upsampled_dataset,
        tokenizer=base_tokenizer,
        optimizers=(optimizer_extension, None)
    )
    
    trainer_final.train()
    
    trainer_final.save_model("final_mosaic_bert")
    base_tokenizer.save_pretrained("final_mosaic_bert")
    wandb.finish()

if __name__ == "__main__":
    main()
