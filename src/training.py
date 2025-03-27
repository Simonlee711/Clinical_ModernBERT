import math
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback
import wandb

def calculate_training_parameters(dataset, base_batch_size=128, base_epochs=50, memory_limit_gb=32, tokens_per_sample=512):
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
    
    steps_per_epoch = (total_samples + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(int(0.1 * total_steps), 100)
    
    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "tokens_per_sample": tokens_per_sample
    }

class DynamicDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, initial_mlm_probability=0.30, final_mlm_probability=0.15, total_epochs=50, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=initial_mlm_probability, **kwargs)
        self.initial_mlm_probability = initial_mlm_probability
        self.final_mlm_probability = final_mlm_probability
        self.total_epochs = total_epochs

    def update_epoch(self, current_epoch):
        fraction = current_epoch / max(1, self.total_epochs - 1)
        new_prob = self.initial_mlm_probability - ((self.initial_mlm_probability - self.final_mlm_probability) * fraction)
        self.mlm_probability = max(min(new_prob, 1.0), 0.0)

class DynamicMaskingCallback(TrainerCallback):
    def __init__(self, data_collator):
        self.data_collator = data_collator

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = state.epoch
        self.data_collator.update_epoch(current_epoch)
        wandb.log({"mlm_probability": self.data_collator.mlm_probability})
        return control

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

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
        loss = loss / self.args.gradient_accumulation_steps
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None]
        if parameters:
            total_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in parameters]), 2).item()
        wandb.log({"gradient_norm": total_norm})
        return loss.detach()

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        
        import time, os
        timestamp = int(time.time())
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        for file in os.listdir(checkpoint_dir):
            print(f"Saved file: {file}")

def create_cosine_lr_scheduler(optimizer, num_training_steps, num_warmup_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
