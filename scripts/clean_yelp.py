import pandas as pd
import re

def clean_text(text):
    """
    Clean a review (same function we used for IMDB)
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

# Load Yelp data
print("Loading Yelp reviews...")
df = pd.read_csv('yelp_reviews.csv')

print(f"Original dataset: {len(df)} reviews")

# Show before
print("\n" + "="*60)
print("BEFORE CLEANING:")
print("="*60)
print(df.iloc[0]['text'][:300])

# Clean
print("\nCleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Show after
print("\n" + "="*60)
print("AFTER CLEANING:")
print("="*60)
print(df.iloc[0]['cleaned_text'][:300])

# Stats
print("\n" + "="*60)
print("CLEANING SUMMARY:")
print("="*60)
print(f"Original avg length: {df['text'].str.len().mean():.0f} characters")
print(f"Cleaned avg length: {df['cleaned_text'].str.len().mean():.0f} characters")

# Save
df[['cleaned_text', 'label']].to_csv('yelp_reviews_cleaned.csv', index=False)
print("\nSaved to: yelp_reviews_cleaned.csv")