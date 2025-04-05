#!/usr/bin/env python
import os
import shutil
import logging
import gc

import torch
import wandb
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset

from src import data, model, training, utils


def main():
    os.environ["HF_HOME"] = "./home"
    os.environ["HF_HUB_CACHE"] = "./home/hf_cache"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    wandb.login(key=[WANDB_API_KEY])
    wandb.init(
        project="BioClinical ModernBERT",
        name="clinical-text-pretraining",
        config={
            "model_type": "MosaicBERT",
            "context_length": 8192,
            "initial_learning_rate": 3e-4,
            "batch_size_initial": utils.get_dynamic_batch_size(),
            "mlm_probability": 0.30,
            "warmup_ratio": 0.1,
            "epochs_initial": 50,
            "gradient_accumulation_steps": 4
        }
    )

    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", None)
    utils.log_memory_usage(logging)
    os.makedirs("checkpoints_mosaic_bert_smaller_pretrain", exist_ok=True)
    os.makedirs("final_mosaic_bert_smaller_pretrain", exist_ok=True)

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

    data_collator = training.DynamicDataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        initial_mlm_probability=0.30,
        final_mlm_probability=0.15,
        total_epochs=wandb.config.epochs_initial
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
        save_steps=10000,
        logging_steps=10000,
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

    final_model_path = "final_mosaic_bert_smaller_pretrain"
    shutil.rmtree(final_model_path, ignore_errors=True)
    os.makedirs(final_model_path)

    trainer.save_model(final_model_path)
    base_tokenizer.save_pretrained(final_model_path)

    utils.cleanup_memory()
    utils.log_memory_usage(logging)
    wandb.finish()


if __name__ == "__main__":
    main()
