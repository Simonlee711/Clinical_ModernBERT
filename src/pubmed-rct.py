import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import OneHotEncoder

# Configuration
MODEL_NAME = 'bert-base-uncased'
DATASET_NAME = 'pietrolesci/pubmed-200k-rct'
MAX_LENGTH = 512
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3  # Reduced epochs for faster testing
RANDOM_SEED = 42
NUM_BOOTSTRAPS = 100  # Reduced bootstraps
CONFIDENCE_LEVEL = 0.95
SAMPLE_FRACTION = 0.001  # Sample 1% of the data

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load dataset
def load_pubmed_dataset():
    dataset = load_dataset(DATASET_NAME)
    
    # Print available columns for debugging
    print("Available columns:", dataset['train'].column_names)
    
    # Use 'labels' instead of 'label'
    X = dataset['train']['text']
    y = dataset['train']['labels']
    
    # Sample 1% of the data
    sample_size = int(len(X) * SAMPLE_FRACTION)
    indices = np.random.choice(len(X), sample_size, replace=False)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    
    # Print unique labels for verification
    print("Unique labels:", set(y))
    
    # Ensure labels are integers
    unique_labels = list(set(y))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = [label_map[label] for label in y]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, len(unique_labels)

# Tokenization and encoding
def prepare_data(X_train, X_test, y_train, y_test, model_name, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize and encode training data
    train_encodings = tokenizer(
        list(X_train), 
        truncation=True, 
        padding=True, 
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Tokenize and encode test data
    test_encodings = tokenizer(
        list(X_test), 
        truncation=True, 
        padding=True, 
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Convert to tensor datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(y_train)
    )
    
    test_dataset = TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(y_test)
    )
    
    return train_dataset, test_dataset

# Linear Probing Model
class LinearProbingClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        
        # Freeze base model parameters
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings (using [CLS] token)
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

# Training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    # Progress bar for training
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    
    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(train_loader)

# Evaluation function with bootstrapping for confidence intervals
def evaluate_model(model, test_loader, device, num_classes, num_bootstraps=1000, confidence_level=0.95):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Progress bar for evaluation
    progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate initial metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    
    # One-hot encode for AUROC and AUPRC
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_true_onehot = onehot_encoder.fit_transform(
        np.array(all_labels).reshape(-1, 1)
    )
    
    # Compute softmax probabilities
    model.eval()
    y_pred_proba = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, _ = batch
            outputs = model(input_ids.to(device), attention_mask.to(device))
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred_proba.extend(proba)
    y_pred_proba = np.array(y_pred_proba)
    
    # AUROC and AUPRC
    auroc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr')
    auprc = average_precision_score(y_true_onehot, y_pred_proba, average='weighted')
    
    # Bootstrapping for confidence intervals
    bootstrap_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auroc': [],
        'auprc': []
    }
    
    for _ in tqdm(range(num_bootstraps), desc="Bootstrapping"):
        # Sample with replacement
        indices = np.random.randint(0, len(all_labels), len(all_labels))
        boot_labels = np.array(all_labels)[indices]
        boot_preds = np.array(all_preds)[indices]
        
        # Compute metrics for this bootstrap sample
        bootstrap_metrics['accuracy'].append(accuracy_score(boot_labels, boot_preds))
        
        boot_precision, boot_recall, boot_f1, _ = precision_recall_fscore_support(
            boot_labels, boot_preds, average='weighted'
        )
        bootstrap_metrics['precision'].append(boot_precision)
        bootstrap_metrics['recall'].append(boot_recall)
        bootstrap_metrics['f1_score'].append(boot_f1)
        
        # Compute bootstrap probabilities (random sampling)
        boot_true_onehot = onehot_encoder.transform(boot_labels.reshape(-1, 1))
        boot_pred_proba = y_pred_proba[indices]
        
        bootstrap_metrics['auroc'].append(
            roc_auc_score(boot_true_onehot, boot_pred_proba, multi_class='ovr')
        )
        bootstrap_metrics['auprc'].append(
            average_precision_score(boot_true_onehot, boot_pred_proba, average='weighted')
        )
    
    # Compute confidence intervals
    confidence_intervals = {}
    for metric, values in bootstrap_metrics.items():
        # Compute percentile-based confidence interval
        lower_bound = np.percentile(values, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(values, (1 + confidence_level) / 2 * 100)
        
        confidence_intervals[metric] = {
            'point_estimate': locals()[metric],
            'ci_lower': lower_bound,
            'ci_upper': upper_bound
        }
    
    return confidence_intervals

def main():
    # Device configuration with explicit CUDA selection
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure CUDA is being used efficiently
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, num_classes = load_pubmed_dataset()
    train_dataset, test_dataset = prepare_data(
        X_train, X_test, y_train, y_test, MODEL_NAME, MAX_LENGTH
    )
    
    # Create data loaders with pin_memory for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
    
    # Load pre-trained model
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    
    # Create linear probing model
    model = LinearProbingClassifier(base_model, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    # Training loop with overall progress bar
    print("Starting Training...")
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch"):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}')
    
    # Evaluate with confidence intervals
    print("\nEvaluating model...")
    metrics = evaluate_model(
        model, 
        test_loader, 
        device, 
        num_classes,
        num_bootstraps=NUM_BOOTSTRAPS, 
        confidence_level=CONFIDENCE_LEVEL
    )
    
    # Print results with confidence intervals
    print(f"\nEvaluation Metrics (with {CONFIDENCE_LEVEL*100}% Confidence Intervals):")
    for metric, stats in metrics.items():
        print(f"{metric}: {stats['point_estimate']:.4f} "
              f"({stats['ci_lower']:.4f} - {stats['ci_upper']:.4f})")

if __name__ == "__main__":
    main()
