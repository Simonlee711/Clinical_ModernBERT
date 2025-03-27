#!/usr/bin/env python
import os
import time
import shutil
import logging
import gc

import torch
import wandb
from transformers import AutoTokenizer, TrainingArguments

from datasets import Dataset

from src import data, model, training, utils

def main():
    # Environment setup
    os.environ["HF_HOME"] = "./home"
    os.environ["HF_HUB_CACHE"] = "./home/hf_cache"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    
    # Initialize WandB
    wandb.login(key="0d3cef273ac07263f8b9035513b8693a26308dce")
    wandb.init(
        project="BioClinical ModernBERT_e2e2", 
        name="clinical-text-pretraining",
        config={
            "model_type": "MosaicBERT",
            "context_length": 8192,
            "initial_learning_rate": 3e-4,
            "extended_learning_rate": 1e-5,
            "batch_size_initial": utils.get_dynamic_batch_size(),
            "batch_size_extended": utils.get_dynamic_batch_size(initial_batch_size=128),
            "mlm_probability": 0.30,
            "warmup_ratio": 0.1,
            "epochs_initial": 50,
            "epochs_extended": 50,
            "gradient_accumulation_steps": 4
        }
    )
    
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", None)
    utils.log_memory_usage(logging)
    os.makedirs("checkpoints_mosaic_bert_smaller_pretrain2", exist_ok=True)
    os.makedirs("final_mosaic_bert_smaller_pretrain2", exist_ok=True)
    
    # Data Loading and Tokenization
    df = data.load_data()
    logging.info(f"Using a subset of data: {df.shape[0]} records")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"DataFrame memory usage: {mem_mb:.2f} MB")
    
    df = df.reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    
    base_tokenizer = AutoTokenizer.from_pretrained("./models/ModernBERT-base/")
    
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
    
    config = model.MosaicBertConfig.from_pretrained("./models/ModernBERT-base/")
    config.vocab_size = len(base_tokenizer)
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    config.flash_attention = True
    config.alibi = True
    config.gated_linear_units = True
    config.use_low_precision_layernorm = True
    config.rope_theta = 10000
    config.context_length = 1024
    
    mbert_model = model.MosaicBertForMaskedLM.from_pretrained("./models/ModernBERT-base/", config=config)
    mbert_model.resize_token_embeddings(len(base_tokenizer))
    
    # Dynamic data collator and training parameter calculation
    initial_epochs = 50
    data_collator = training.DynamicDataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        initial_mlm_probability=0.30,
        final_mlm_probability=0.15,
        total_epochs=initial_epochs
    )
    
    training_params = training.calculate_training_parameters(dataset)
    
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
            "params": [p for n, p in mbert_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in mbert_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    from src.model import StableAdamW
    optimizer = StableAdamW(
        optimizer_grouped_parameters,
        lr=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        clipping_threshold=1.0
    )
    
    trainer = training.CustomTrainer(
        model=mbert_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=base_tokenizer,
        optimizers=(optimizer, None)
    )
    
    trainer.add_callback(training.DynamicMaskingCallback(data_collator))
    
    lr_scheduler = training.create_cosine_lr_scheduler(
        trainer.optimizer, 
        num_training_steps=num_training_steps, 
        num_warmup_steps=num_warmup_steps
    )
    trainer.lr_scheduler = lr_scheduler
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    logging.info("Clearing GPU memory after phase one")
    del trainer
    torch.cuda.empty_cache()
    
    # Context extension phase
    logging.info("Starting context length extension phase")
    config.rope_theta = 160000
    config.context_length = 2048
    
    extended_dataset = data.create_extended_context_dataset(dataset, max_length=2048)
    
    def tokenize_extended_function(examples):
        return base_tokenizer(
            examples["clinical_text"], 
            truncation=True, 
            padding="max_length", 
            max_length=2048
        )
    
    tokenized_extended_dataset = extended_dataset.map(
        tokenize_extended_function, 
        batched=True, 
        num_proc=20, 
        remove_columns=["clinical_text"]
    )
    tokenized_extended_dataset = tokenized_extended_dataset.with_format("torch")
    
    extended_training_params = training.calculate_training_parameters(
        extended_dataset, 
        base_batch_size=16, 
        memory_limit_gb=32, 
        tokens_per_sample=2048
    )
    
    training_args_extension = TrainingArguments(
        output_dir="checkpoints_mosaic_bert_extended_smaller_pretrain",
        overwrite_output_dir=True,
        run_name="mosaicbert_context_extension",
        num_train_epochs=extended_training_params["epochs"],
        per_device_train_batch_size=extended_training_params["batch_size"],
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=20000,
        logging_steps=20000,
        learning_rate=1e-5,
        bf16=True,
        report_to=["wandb"],
        save_total_limit=None,
        remove_unused_columns=False,
        local_rank=-1,
        dataloader_num_workers=20,
        fp16_backend="auto",
        no_cuda=False,
    )
    
    num_extended_steps = extended_training_params["total_steps"]
    num_extended_warmup_steps = extended_training_params["warmup_steps"]
    
    optimizer_extension = StableAdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        clipping_threshold=1.0
    )
    
    trainer_extension = training.CustomTrainer(
        model=mbert_model,
        args=training_args_extension,
        data_collator=data_collator,
        train_dataset=tokenized_extended_dataset,
        tokenizer=base_tokenizer,
        optimizers=(optimizer_extension, None)
    )
    
    lr_scheduler_extension = training.create_cosine_lr_scheduler(
        trainer_extension.optimizer, 
        num_training_steps=num_extended_steps, 
        num_warmup_steps=num_extended_warmup_steps
    )
    trainer_extension.lr_scheduler = lr_scheduler_extension
    
    trainer_extension.train()
    
    final_model_path = "final_mosaic_bert_smaller_pretrain2"
    shutil.rmtree(final_model_path, ignore_errors=True)
    os.makedirs(final_model_path)
    
    trainer_extension.save_model(final_model_path)
    base_tokenizer.save_pretrained(final_model_path)
    
    cleanup_memory()
    utils.log_memory_usage(logging)
    wandb.finish()

if __name__ == "__main__":
    main()
