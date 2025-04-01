import os
import json
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, DistilBertTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_and_process_data():
    # Load JSON data from the specified bucket and file.
    bucket_name = '/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/mimic/mimic-iv-ed-2.2/mimic-iv-ed-2.2'
    file_content = 'text_repr.json'
    with open(os.path.join(bucket_name, file_content)) as f:
        json_content = json.load(f)
    
    # Create a DataFrame from the JSON content (transposed).
    df = pd.DataFrame(json_content).T

    # Process the columns as specified.
    df['eddischarge'] = [1 if 'admitted' in s.lower() else 0 for s in df['eddischarge']]
    df['medrecon'] = df['medrecon'].fillna("The patient was previously not taking any medications.")
    df['pyxis'] = df['pyxis'].fillna("The patient did not receive any medications.")
    df['vitals'] = df['vitals'].fillna("The patient had no vitals recorded")
    df['codes'] = df['codes'].fillna("The patient received no diagnostic codes")
    
    df = df.drop("admission", axis=1)
    df = df.drop("discharge", axis=1)
    df = df.drop("eddischarge_category", axis=1)
    
    # Create the "info" column by concatenating the relevant text fields.
    df["info"] = df['arrival'] + df["codes"] + df["triage"] + df["vitals"] + df["pyxis"] + df["medrecon"]
    
    df = df.drop("arrival", axis=1)
    df = df.drop("codes", axis=1)
    df = df.drop("triage", axis=1)
    df = df.drop("vitals", axis=1)
    df = df.drop("pyxis", axis=1)
    df = df.drop("medrecon", axis=1)
    
    # Rearrange columns so that 'eddischarge' appears at the end.
    df = df[[col for col in df.columns if col != 'eddischarge'] + ['eddischarge']]
    
    return df

def initialize_tokenizer_for_model(model_id, model_type):
    """Initialize the appropriate tokenizer based on model type"""
    try:
        if 'distil' in model_type.lower():
            tokenizer = DistilBertTokenizer.from_pretrained(model_id)
            print(f"Successfully loaded DistilBertTokenizer for {model_type}")
        else:
            tokenizer = BertTokenizer.from_pretrained(model_id)
            print(f"Successfully loaded BertTokenizer for {model_type}")
    except Exception as e:
        print(f"Specific tokenizer failed: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            print(f"Successfully loaded tokenizer with AutoTokenizer")
        except Exception as e:
            print(f"AutoTokenizer failed: {e}")
            print("Using bert-base-uncased tokenizer as fallback...")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

def pre_tokenize_data(tokenizer, texts, max_length=512):
    """Pre-tokenize all texts to avoid tokenization during embedding generation"""
    tokenized_texts = []
    for text in tqdm(texts, desc="Pre-tokenizing", leave=False):
        encoded = tokenizer(text, return_tensors="pt", padding="max_length", 
                          truncation=True, max_length=max_length)
        tokenized_texts.append(encoded)
    return tokenized_texts

def generate_embeddings_from_tokenized(model, tokenized_texts, batch_size=64, device=torch.device("cpu")):
    """Generate embeddings from pre-tokenized texts"""
    embeddings_list = []
    for i in tqdm(range(0, len(tokenized_texts), batch_size), desc="Processing batches", leave=False):
        batch = tokenized_texts[i:i+batch_size]
        # Combine batch into a single tensor dictionary
        batch_dict = {k: torch.cat([item[k] for item in batch], dim=0) for k in batch[0].keys()}
        
        # Remove token_type_ids if the model doesn't need them
        if 'token_type_ids' in batch_dict and 'distil' in str(type(model)).lower():
            batch_dict.pop('token_type_ids')
            
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        outputs = model(**batch_dict)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
        embeddings_list.append(batch_embeddings)
    return torch.cat(embeddings_list, dim=0)

def generate_embeddings(model, tokenizer, texts, batch_size=64, device=torch.device("cpu")):
    """Handle different model types appropriately"""
    embeddings_list = []
    # Wrap the batch loop with tqdm to display progress.
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", leave=False):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Remove token_type_ids if the model doesn't need them
        if 'token_type_ids' in encoded_input and 'distil' in str(type(model)).lower():
            encoded_input.pop('token_type_ids')
            
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        outputs = model(**encoded_input)
        # Extract the [CLS] token embeddings, detach and move them to CPU.
        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
        embeddings_list.append(batch_embeddings)
    return torch.cat(embeddings_list, dim=0)

def main():
    # Load and process the data.
    df = load_and_process_data()
    texts = df["info"].tolist()
    
    # Limit to 100k examples for the full benchmark.
    if len(texts) > 100000:
        texts_full = texts[:100000]
    else:
        texts_full = texts
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Define the models to compare.
    models_to_test = {
        "Distil-BERT": "/opt/data/commonfilesharePHI/slee/GeneratEHR/distil-bert",
        "BioClinicalBERT": "/opt/data/commonfilesharePHI/slee/GeneratEHR/clinicalBERT-emily",
        "Clinical ModernBERT": "./checkpoints_mosaic_bert_smaller_pretrain3/checkpoint-120000/checkpoint-1743062013"
    }
    
    full_elapsed_times = {}
    
    # Benchmark full encoding time for each model.
    for model_name, model_id in models_to_test.items():
        print(f"Full benchmark for {model_name} on {len(texts_full)} examples...")
        
        # Debug: Check files in checkpoint directory
        print(f"Files in checkpoint directory: {os.listdir(model_id)}")
        
        # Initialize model first
        model = AutoModel.from_pretrained(model_id)
        model_type = model.__class__.__name__
        model.to(device)
        model.eval()
        
        # Get appropriate tokenizer for this model type
        tokenizer = initialize_tokenizer_for_model(model_id, model_type)
        
        # Test the tokenizer with sample text
        try:
            test_result = tokenizer("Test sentence", return_tensors="pt")
            print("Tokenizer test successful")
            use_standard_method = True
        except Exception as e:
            print(f"Tokenizer test failed: {e}")
            print("Using an alternative approach with pre-tokenized data...")
            use_standard_method = False
            
            # If direct tokenizer fails, use the base tokenizer and pre-tokenize everything
            base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            print("Pre-tokenizing all texts...")
            tokenized_texts = pre_tokenize_data(base_tokenizer, texts_full)
            
            # Use the method for pre-tokenized data
            print("Generating embeddings from pre-tokenized texts...")
            start_time = time.time()
            _ = generate_embeddings_from_tokenized(model, tokenized_texts, batch_size=64, device=device)
            elapsed = time.time() - start_time
            full_elapsed_times[model_name] = elapsed
            print(f"{model_name} took {elapsed:.2f} seconds for {len(texts_full)} data points.")
            continue
        
        # If we get here, the standard tokenizer is working
        start_time = time.time()
        _ = generate_embeddings(model, tokenizer, texts_full, batch_size=64, device=device)
        elapsed = time.time() - start_time
        full_elapsed_times[model_name] = elapsed
        print(f"{model_name} took {elapsed:.2f} seconds for {len(texts_full)} data points.")
    
    # Create a bar plot for full benchmark times.
    plt.figure(figsize=(8, 6))
    plt.bar(full_elapsed_times.keys(), full_elapsed_times.values())
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.title("Embedding Generation Time for Different Models")
    plt.tight_layout()
    plt.savefig("embedding_times.png")
    plt.show()
    
    # For incremental measurements, work with up to 100k examples.
    max_points = 100000 if len(texts) >= 100000 else len(texts)
    texts_incremental = texts[:max_points]
    # Define the data points (every 10k points).
    data_points = list(range(10000, max_points + 1, 10000))
    
    incremental_times = {}
    
    # Measure encoding time on incremental subsets for each model.
    for model_name, model_id in models_to_test.items():
        print(f"Incremental benchmark for {model_name}...")
        
        # Initialize model and get its type
        model = AutoModel.from_pretrained(model_id)
        model_type = model.__class__.__name__
        model.to(device)
        model.eval()
        
        # Get appropriate tokenizer
        tokenizer = initialize_tokenizer_for_model(model_id, model_type)
        
        # Test tokenizer again
        try:
            test_result = tokenizer("Test sentence", return_tensors="pt")
            use_standard_method = True
        except Exception as e:
            print(f"Tokenizer test failed: {e}")
            use_standard_method = False
            base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        times_list = []
        for n in tqdm(data_points, desc=f"{model_name} incremental", leave=False):
            start_time = time.time()
            
            if use_standard_method:
                _ = generate_embeddings(model, tokenizer, texts_incremental[:n], batch_size=64, device=device)
            else:
                tokenized_texts = pre_tokenize_data(base_tokenizer, texts_incremental[:n])
                _ = generate_embeddings_from_tokenized(model, tokenized_texts, batch_size=64, device=device)
            
            t_elapsed = time.time() - start_time
            times_list.append(t_elapsed)
            print(f"{model_name}: {n} points took {t_elapsed:.2f} seconds")
        
        incremental_times[model_name] = times_list
    
    # Create a dotted line plot to show incremental encoding times.
    plt.figure(figsize=(8, 6))
    for model_name, times_list in incremental_times.items():
        plt.plot(data_points, times_list, linestyle=':', marker='o', label=model_name)
    plt.xlabel("Number of Data Points")
    plt.ylabel("Time (seconds)")
    plt.title("Embedding Generation RunTime")
    plt.legend()
    plt.tight_layout()
    plt.savefig("incremental_encoding_times.png")
    plt.show()

if __name__ == "__main__":
    main()
