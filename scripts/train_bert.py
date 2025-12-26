import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

print("Loading cleaned data...")
df = pd.read_csv('imdb_train_cleaned.csv')
# Use a small subset first (for quick testing)
df_sample = df.sample(n=1000, random_state=42)

print(f"Using {len(df_sample)} samples for training")

# Split into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_sample['cleaned_text'].tolist(),
    df_sample['label'].tolist(),
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Load tokenizer
print("\nLoading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
print("Tokenizing data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Create PyTorch dataset
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

# Load pre-trained model
print("\nLoading pre-trained BERT model...")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
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

# Train!
print("\n" + "="*60)
print("Starting training...")
print("="*60)
trainer.train()

print("\nTraining complete!")
print("\nEvaluating on validation set...")
results = trainer.evaluate()

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
for key, value in results.items():
    print(f"{key}: {value:.4f}")