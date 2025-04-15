"""
Fine-tune DistilBERT for text classification
"""
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset and tokenizer
dataset = load_dataset("imdb")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Training
training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["test"])
trainer.train()
