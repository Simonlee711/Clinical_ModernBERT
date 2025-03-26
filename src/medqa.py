import numpy as np
import logging
import sys
from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForMultipleChoice,
    EarlyStoppingCallback
)
from datasets import load_dataset

def load_and_prepare_dataset(sample_percentage=0.1):
    """
    Load the BigBio MedQA dataset and prepare train/validation splits with sampling.
    
    Args:
        sample_percentage (float): Percentage of data to use (default 10%)
    
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    try:
        dataset = load_dataset("bigbio/med_qa")
        
        # Create train and validation splits
        if "train" in dataset:
            # If train split exists, create validation split
            train_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            train_dataset = train_split["train"]
            eval_dataset = train_split["test"]
        else:
            # Fallback to splitting entire dataset
            train_split = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = train_split["train"]
            eval_dataset = train_split["test"]
        
        # Sample a percentage of the data
        train_dataset = train_dataset.shuffle(seed=42).select(
            range(int(len(train_dataset) * sample_percentage))
        )
        eval_dataset = eval_dataset.shuffle(seed=42).select(
            range(int(len(eval_dataset) * sample_percentage))
        )
        
        logging.info(f"Using {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")
        
        return train_dataset, eval_dataset
    
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

def preprocess_function(example, tokenizer):
    """
    Preprocess each example by tokenizing question and options.
    
    Args:
        example (dict): A single dataset example
        tokenizer: Tokenizer to use for encoding
    
    Returns:
        dict: Tokenized and formatted example
    """
    question = example["question"]
    
    # Extract options and their keys
    options = [option['value'] for option in example['options']]
    option_keys = [option['key'] for option in example['options']]
    
    # Combine the question with each option
    inputs = [f"{question} {option}" for option in options]
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    
    # Find the correct label index
    correct_label_key = example['answer_idx']
    correct_label = option_keys.index(correct_label_key)
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": correct_label,
    }

def setup_model_and_tokenizer():
    """
    Load pretrained model and tokenizer.
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
        
        return model, tokenizer
    
    except Exception as e:
        logging.error(f"Error setting up model and tokenizer: {e}")
        sys.exit(1)

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Prediction results from trainer
    
    Returns:
        dict: Accuracy metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def main():
    """
    Main function to run medical QA fine-tuning.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("medqa_training.log")
        ]
    )
    
    # Load and prepare dataset (10% sampling)
    train_dataset, eval_dataset = load_and_prepare_dataset(sample_percentage=0.1)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize the datasets
    tokenized_train = train_dataset.map(
        lambda example: preprocess_function(example, tokenizer), 
        batched=False,
        remove_columns=train_dataset.column_names
    )
    tokenized_eval = eval_dataset.map(
        lambda example: preprocess_function(example, tokenizer), 
        batched=False,
        remove_columns=eval_dataset.column_names
    )
    
    # Define training arguments with early stopping and validation tracking
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",        # Save model at the end of each epoch
        logging_strategy="epoch",     # Log at the end of each epoch
        learning_rate=3e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,           # Reduced epochs for quick testing
        weight_decay=0.01,
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Use validation loss to determine best model
    )
    
    # Data collator for multiple-choice inputs
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    
    # Initialize Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Stop if no improvement for 3 epochs
                early_stopping_threshold=0.001  # Minimum change to consider as improvement
            )
        ]
    )
    
    # Fine-tune the model
    logging.info("Starting model fine-tuning...")
    trainer.train()
    
    # Evaluate the final model
    logging.info("Starting final model evaluation...")
    eval_result = trainer.evaluate()
    logging.info("Final evaluation results: %s", eval_result)
    
    # Save the final model
    trainer.save_model("./final_medical_qa_model")
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    main()
