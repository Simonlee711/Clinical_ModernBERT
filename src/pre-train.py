#!/usr/bin/env python
import os
import re
import glob
import gzip
import logging
import random
import math
import shutil
import time

import pandas as pd
import numpy as np
import torch
import swifter
from tqdm import tqdm
from lxml import etree
import wandb

from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    BertConfig, 
    BertForMaskedLM,
    TrainerCallback
)
from datasets import Dataset
from typing import Optional

from concurrent.futures import ProcessPoolExecutor
import torch.nn as nn

# Explicit memory management and monitoring
import gc
import psutil
import GPUtil

# Set environment variables for efficient processing
os.environ["HF_HOME"] = "./home"
os.environ["HF_HUB_CACHE"] = "./home/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Memory logging function
def log_memory_usage(logger=None):
    # CPU Memory
    cpu_memory_percent = psutil.virtual_memory().percent
    if logger:
        logger.info(f"Total RAM Used: {cpu_memory_percent}%")
    else:
        print(f"Total RAM Used: {cpu_memory_percent}%")
    
    # GPU Memory
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if logger:
            logger.info(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        else:
            print(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB")

# Explicit memory cleanup function
def cleanup_memory():
    gc.collect()  # Python garbage collection
    torch.cuda.empty_cache()  # Clear CUDA memory

# Memory-efficient data type reduction
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

# Dynamically adjust batch size based on available memory
def get_dynamic_batch_size(initial_batch_size=128, min_batch_size=32):
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        
        # Memory-based batch size adjustment
        if available_memory < 10 * 1024 * 1024 * 1024:  # Less than 10GB
            return max(initial_batch_size // 4, min_batch_size)
        elif available_memory < 20 * 1024 * 1024 * 1024:  # Less than 20GB
            return initial_batch_size // 2
        return initial_batch_size
    except Exception:
        return initial_batch_size

# Low Precision Layer Norm for memory efficiency
class LowPrecisionLayerNorm(nn.LayerNorm):
    def forward(self, input):
        orig_dtype = input.dtype
        if input.dtype == torch.float32:
            input = input.to(torch.bfloat16)
        out = super().forward(input)
        return out.to(orig_dtype)

# Existing configuration and model classes with some modifications
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

def calculate_training_parameters(dataset, 
                                  base_batch_size=128, 
                                  base_epochs=50, 
                                  memory_limit_gb=32, 
                                  tokens_per_sample=512):
    """
    Dynamically calculate training parameters based on dataset characteristics
    
    Args:
        dataset (Dataset): Hugging Face dataset
        base_batch_size (int): Initial batch size to start with
        base_epochs (int): Default number of epochs
        memory_limit_gb (int): Maximum GPU memory limit
        tokens_per_sample (int): Average number of tokens per sample
    
    Returns:
        dict: Computed training parameters
    """
    total_samples = len(dataset)
    total_tokens = total_samples * tokens_per_sample
    
    try:
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_gpu_memory = total_gpu_memory * 0.9
        
        if available_gpu_memory < memory_limit_gb:
            batch_size = max(base_batch_size // 2, 32)
        else:
            batch_size = base_batch_size
    except:
        batch_size = base_batch_size
    
    if total_samples < 10000:
        epochs = min(base_epochs * 2, 100)
    elif total_samples < 100000:
        epochs = base_epochs
    elif total_samples < 1000000:
        epochs = max(base_epochs // 2, 10)
    else:
        epochs = max(base_epochs // 4, 5)
    
    steps_per_epoch = math.ceil(total_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(int(0.1 * total_steps), 100)
    
    logging.info(f"Dataset Analysis:")
    logging.info(f"Total Samples: {total_samples}")
    logging.info(f"Total Tokens: {total_tokens:,}")
    logging.info(f"Computed Batch Size: {batch_size}")
    logging.info(f"Computed Epochs: {epochs}")
    logging.info(f"Total Training Steps: {total_steps}")
    logging.info(f"Warmup Steps: {warmup_steps}")
    
    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "tokens_per_sample": tokens_per_sample
    }

# Custom dynamic data collator that updates its masking probability.
class DynamicDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, initial_mlm_probability=0.30, final_mlm_probability=0.15, total_epochs=50, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=initial_mlm_probability, **kwargs)
        self.initial_mlm_probability = initial_mlm_probability
        self.final_mlm_probability = final_mlm_probability
        self.total_epochs = total_epochs

    def update_epoch(self, current_epoch):
        # Linear schedule from initial to final over total_epochs
        fraction = current_epoch / max(1, self.total_epochs - 1)
        new_prob = self.initial_mlm_probability - ((self.initial_mlm_probability - self.final_mlm_probability) * fraction)
        # Clamp the new probability to be within [0, 1]
        new_prob = max(min(new_prob, 1.0), 0.0)
        self.mlm_probability = new_prob


# Callback to update the collator at each epoch
class DynamicMaskingCallback(TrainerCallback):
    def __init__(self, data_collator):
        self.data_collator = data_collator

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = state.epoch
        self.data_collator.update_epoch(current_epoch)
        wandb.log({"mlm_probability": self.data_collator.mlm_probability})
        return control

# Custom Trainer with enhanced accuracy tracking, gradient norm logging, and modified training step for gradient accumulation.
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        # Calculate MLM accuracy
        predictions = outputs.logits
        labels = inputs.get("labels")
        masked_lm_mask = (labels != -100)
        masked_predictions = predictions[masked_lm_mask]
        masked_labels = labels[masked_lm_mask]

        top1_predictions = masked_predictions.argmax(dim=-1)
        top1_accuracy = (top1_predictions == masked_labels).float().mean().item()

        top5_predictions = torch.topk(masked_predictions, k=5, dim=-1).indices
        top5_accuracy = torch.any(top5_predictions == masked_labels.unsqueeze(1), dim=1).float().mean().item()

        top10_predictions = torch.topk(masked_predictions, k=10, dim=-1).indices
        top10_accuracy = torch.any(top10_predictions == masked_labels.unsqueeze(1), dim=1).float().mean().item()

        top25_predictions = torch.topk(masked_predictions, k=25, dim=-1).indices
        top25_accuracy = torch.any(top25_predictions == masked_labels.unsqueeze(1), dim=1).float().mean().item()

        wandb.log({
            "mlm_loss": loss.item(),
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "top10_accuracy": top10_accuracy,
            "top25_accuracy": top25_accuracy,
        })
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        # Adjust loss for gradient accumulation if applicable.
        loss = loss / self.args.gradient_accumulation_steps
        self.optimizer.zero_grad()
        loss.backward()
        # Compute gradient norm for logging.
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None]
        if parameters:
            total_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in parameters]), 2).item()
        wandb.log({"gradient_norm": total_norm})
        return loss.detach()

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        
        timestamp = int(time.time())
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{timestamp}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        logging.info(f"Saved comprehensive checkpoint to {checkpoint_dir}")
        for file in os.listdir(checkpoint_dir):
            logging.info(f"Saved file: {file}")


def create_cosine_lr_scheduler(optimizer, num_training_steps, num_warmup_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_csv_and_clean(path, label, chunk_size=100000):
    logging.info(f"Loading {label} notes from {path}")
    dfs = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        columns_to_drop = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime"]
        chunk = chunk.drop(columns=[col for col in columns_to_drop if col in chunk.columns])
        chunk["clinical_text"] = chunk["text"].swifter.apply(clean_text)
        chunk = chunk.drop(columns="text")
        dfs.append(chunk)
    df = pd.concat(dfs)
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
    with open(file, 'rb') as f:
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
    discharge_path = "./data/mimic-note/discharge.csv.gz"
    radiology_path = "./data/mimic-note/radiology.csv.gz"
    pubmed_path_glob = "./data/pubmed/*.xml"
    discharge = load_csv_and_clean(discharge_path, label="discharge")
    radiology = load_csv_and_clean(radiology_path, label="radiology")
    pubmed = load_pubmed_and_clean(pubmed_path_glob)
    icd_code = pd.read_csv("./data/coded/icd_codes.csv")
    procedure_code = pd.read_csv("./data/coded/icd_procedures.csv")
    hcpcs_code = pd.read_csv("./data/coded/hcpcs_codes.csv")
    icd_code["clinical_text"] = icd_code["text"]
    procedure_code["clinical_text"] = procedure_code["text"]
    hcpcs_code["clinical_text"] = hcpcs_code["text"]
    
    icd_code = icd_code[["clinical_text"]]
    procedure_code = procedure_code[["clinical_text"]]
    hcpcs_code = hcpcs_code[["clinical_text"]]
    
    discharge = discharge[["clinical_text"]]
    radiology = radiology[["clinical_text"]]
    pubmed = pubmed[["clinical_text"]]
    #combined = pd.concat([ icd_code, procedure_code, hcpcs_code], ignore_index=True)
    combined = pd.concat([discharge, radiology, pubmed, icd_code, procedure_code, hcpcs_code], ignore_index=True) # discharge, radiology, pubmed,
    logging.info(f"Final combined dataset shape: {combined.shape}")
    #combined = combined.sample(100)
    return combined

wandb.login(key=[WANDB_API_KEY])
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
base_tokenizer = AutoTokenizer.from_pretrained("./models/ModernBERT-base/")

def main():
    gc.collect()
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", None)

    wandb.init(
        project="BioClinical ModernBERT",
        name="clinical-text-pretraining",
        config={
            "model_type": "MosaicBERT",
            "context_length": 8192,
            "initial_learning_rate": 3e-4,
            "batch_size_initial": get_dynamic_batch_size(),
            "mlm_probability": 0.30,
            "warmup_ratio": 0.1,
            "epochs_initial": 50,
            "gradient_accumulation_steps": 4
        }
    )

    def save_comprehensive_checkpoint(trainer, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer.model.save_pretrained(checkpoint_dir)
        trainer.tokenizer.save_pretrained(checkpoint_dir)
        logging.info(f"Saved comprehensive checkpoint to {checkpoint_dir}")
        for file in os.listdir(checkpoint_dir):
            logging.info(f"Saved file: {file}")

    log_memory_usage(logging)
    os.makedirs("checkpoints_mosaic_bert_smaller_pretrain", exist_ok=True)
    os.makedirs("final_mosaic_bert_smaller_pretrain", exist_ok=True)

    df = load_data()
    logging.info(f"Using a subset of data: {df.shape[0]} records")
    avg_len = df["clinical_text"].swifter.apply(lambda x: len(x.split())).mean()
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"Average token count: {avg_len:.2f}")
    logging.info(f"DataFrame memory usage: {mem_mb:.2f} MB")

    df = df.reset_index(drop=True)
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return base_tokenizer(
            examples["clinical_text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors='pt'
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=20,
        remove_columns=["clinical_text"]
    )
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

    config = MosaicBertConfig.from_pretrained("./models/ModernBERT-base")
    config.vocab_size = len(base_tokenizer)
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    config.flash_attention = True
    config.alibi = True
    config.gated_linear_units = True
    config.use_low_precision_layernorm = True
    config.rope_theta = 10000
    config.context_length = 1024

    model = MosaicBertForMaskedLM.from_pretrained("./models/ModernBERT-base", config=config)
    model.resize_token_embeddings(len(base_tokenizer))

    data_collator = DynamicDataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        initial_mlm_probability=0.30,
        final_mlm_probability=0.15,
        total_epochs=wandb.config.epochs_initial
    )

    training_params = calculate_training_parameters(dataset)

    training_args = TrainingArguments(
        output_dir="checkpoints_mosaic_bert_smaller_pretrain",
        overwrite_output_dir=True,
        run_name="mosaicbert_pretrain",
        num_train_epochs=training_params["epochs"],
        per_device_train_batch_size=training_params["batch_size"],
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=20000,
        logging_steps=20000,
        learning_rate=3e-4,
        bf16=True,
        report_to=["wandb"],
        save_total_limit=None,
        remove_unused_columns=False,
        local_rank=-1,
        dataloader_num_workers=20,
        fp16_backend="auto",
        no_cuda=False,
        optim="adamw_torch"
    )

    num_training_steps = len(tokenized_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
    num_warmup_steps = int(0.1 * num_training_steps)

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
        lr=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        clipping_threshold=1.0
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=base_tokenizer,
        optimizers=(optimizer, None)
    )

    trainer.add_callback(DynamicMaskingCallback(data_collator))

    lr_scheduler = create_cosine_lr_scheduler(
        trainer.optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )
    trainer.lr_scheduler = lr_scheduler

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logging.info("Saving final model after pretraining")
    trainer.save_model("final_mosaic_bert_smaller_pretrain")
    base_tokenizer.save_pretrained("final_mosaic_bert_smaller_pretrain")

    cleanup_memory()
    log_memory_usage(logging)
    wandb.finish()

