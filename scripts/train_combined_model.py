import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

print("Loading combined dataset...")
df = pd.read_csv('combined_dataset.csv')

# Use 5,000 samples for faster training (you can increase later)
# Stratify ensures we get balanced samples from both sources
df_sample = df.groupby(['source', 'label'], group_keys=False).apply(
    lambda x: x.sample(min(len(x), 625), random_state=42)
)

print(f"Using {len(df_sample)} samples for training")
print(f"\nBreakdown:")
print(df_sample.groupby(['source', 'label']).size())

# Split into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_sample['cleaned_text'].tolist(),
    df_sample['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_sample['label']
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize
print("Tokenizing...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Create dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Load model
print("\nLoading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_combined',
    num_train_epochs=3,  # 3 epochs for better learning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs_combined',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
print("\n" + "="*60)
print("TRAINING ON COMBINED DATASET")
print("="*60)
trainer.train()

print("\nTraining complete!")

# Evaluate
print("\nEvaluating...")
results = trainer.evaluate()

print("\n" + "="*60)
print("FINAL RESULTS:")
print("="*60)
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# Save model
print("\nSaving model...")
model.save_pretrained('./final_model')
tokenizer.save_pretrained('./final_model')
print("Model saved to ./final_model")