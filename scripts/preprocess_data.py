import pandas as pd
import re

def clean_text(text):
    """
    Clean a single review text
    """
    # Remove HTML tags 
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Replace hyphens with spaces 
    text = text.replace('-', ' ')
    
    # Remove special characters and digits (keep letters and basic punctuation)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

# Load data
print("Loading data...")
df = pd.read_csv('imdb_train.csv')

print(f"Original dataset: {len(df)} reviews")

# Show before cleaning
print("\n" + "="*60)
print("BEFORE CLEANING:")
print("="*60)
print(df.iloc[0]['text'][:300])

# Clean the text
print("\nCleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Show after cleaning
print("\n" + "="*60)
print("AFTER CLEANING:")
print("="*60)
print(df.iloc[0]['cleaned_text'][:300])

# Check results
print("\n" + "="*60)
print("CLEANING SUMMARY:")
print("="*60)
print(f"Original avg length: {df['text'].str.len().mean():.0f} characters")
print(f"Cleaned avg length: {df['cleaned_text'].str.len().mean():.0f} characters")

# Save cleaned data
print("\nSaving cleaned data...")
df[['cleaned_text', 'label']].to_csv('imdb_train_cleaned.csv', index=False)
print("Saved to imdb_train_cleaned.csv")

# Clean test data too
print("\n" + "="*60)
print("Now cleaning TEST data...")
print("="*60)

test_df = pd.read_csv('imdb_test.csv')
print(f"Test dataset: {len(test_df)} reviews")

test_df['cleaned_text'] = test_df['text'].apply(clean_text)
test_df[['cleaned_text', 'label']].to_csv('imdb_test_cleaned.csv', index=False)

print("Saved to imdb_test_cleaned.csv")
print("\nAll data cleaned and ready!")