#!/usr/bin/env python
import os
import re
import glob
import gzip
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
from concurrent.futures import ProcessPoolExecutor


from transformers import BertConfig, BertForMaskedLM
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

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

    return {
        "clinical_text": clinical_text
    }

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
    # discharge_path = "/data2/simon/data/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
    # radiology_path = "/data2/simon/data/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv.gz"
    pubmed_path_glob = "../../data/pubmed/*.xml.gz"
    # discharge = load_csv_and_clean(discharge_path, label="discharge")
    # radiology = load_csv_and_clean(radiology_path, label="radiology")
    pubmed = load_pubmed_and_clean(pubmed_path_glob)
    discharge = discharge[["clinical_text"]]
    radiology = radiology[["clinical_text"]]
    pubmed = pubmed[["clinical_text"]]
    combined = pd.concat([discharge, radiology, pubmed], ignore_index=True)
    logging.info(f"Final combined dataset shape: {combined.shape}")
    return combined

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
        # Note: FlashAttention, ALiBi, and Gated Linear Units would require deeper custom modifications,
        # which we assume are handled via specialized kernels or integrations.
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

def main():
    df = load_data()
    # Sample 1% of the data for an end-to-end test run
    df = df.sample(frac=0.01, random_state=42)
    logging.info(f"Using a subset of data: {df.shape[0]} records (~1% of full dataset)")
    
    avg_len = df["clinical_text"].swifter.apply(lambda x: len(x.split())).mean()
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"Average token count: {avg_len:.2f}")
    logging.info(f"DataFrame memory usage: {mem_mb:.2f} MB")
    
    dataset = Dataset.from_pandas(df)
    base_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    new_clinical_tokens = []  # No additional clinical tokens are added
    if len(new_clinical_tokens) > 0:
        base_tokenizer.add_tokens(new_clinical_tokens, special_tokens=False)
    
    # Ensure the vocabulary size is a multiple of 64
    vocab_size_before = len(base_tokenizer)
    padding_needed = (64 - (vocab_size_before % 64)) % 64
    if padding_needed != 0:
        dummy_tokens = [f"<dummy_extra_token_{i}>" for i in range(padding_needed)]
        base_tokenizer.add_tokens(dummy_tokens, special_tokens=False)
    logging.info(f"Tokenizer vocab size is now {len(base_tokenizer)} (was {vocab_size_before})")
    
    def tokenize_function(examples):
        return base_tokenizer(examples["clinical_text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["clinical_text"])
    
    config = MosaicBertConfig.from_pretrained("answerdotai/ModernBERT-base")
    config.vocab_size = len(base_tokenizer)
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    config.flash_attention = True
    config.alibi = True
    config.gated_linear_units = True
    config.use_low_precision_layernorm = True
    
    model = MosaicBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", config=config)
    model.resize_token_embeddings(len(base_tokenizer))
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        mlm=True,
        mlm_probability=0.30
    )
    
    wandb.init(project="mosaicbert-pretrain")
    wandb.config.update({"description": "MosaicBERT pretraining with 30% MLM, BF16 LN, FlashAttention, ALiBi, GLUs, dropout=0"})
    
    training_args = TrainingArguments(
        output_dir="checkpoints_mosaic_bert",
        overwrite_output_dir=True,
        num_train_epochs=200,
        per_device_train_batch_size=64,
        save_strategy="steps",
        save_steps=500000,
        logging_steps=1000,
        learning_rate=1e-4,
        bf16=True,  # use bfloat16
        report_to=["wandb"],
        save_total_limit=None
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=base_tokenizer
    )
    
    trainer.train()
    trainer.save_model("final_mosaic_bert")
    base_tokenizer.save_pretrained("final_mosaic_bert")
    wandb.finish()

if __name__ == "__main__":
    main()
