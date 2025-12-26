import pandas as pd

print("Loading datasets...")

# Load IMDB
imdb_df = pd.read_csv('imdb_train_cleaned.csv')
imdb_df['source'] = 'imdb'  # Track where it came from

# Load Yelp
yelp_df = pd.read_csv('yelp_reviews_cleaned.csv')
yelp_df['source'] = 'yelp'  # Track where it came from

print(f"IMDB reviews: {len(imdb_df):,}")
print(f"Yelp reviews: {len(yelp_df):,}")

# Combine
combined_df = pd.concat([imdb_df, yelp_df], ignore_index=True)

# Shuffle
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset: {len(combined_df):,} reviews")

# Check distribution
print("\n" + "="*60)
print("DATASET BREAKDOWN:")
print("="*60)
print("\nBy source:")
print(combined_df['source'].value_counts())
print("\nBy sentiment:")
print(combined_df['label'].value_counts())
print("\nBy source AND sentiment:")
print(pd.crosstab(combined_df['source'], combined_df['label'], 
                  rownames=['Source'], colnames=['Label']))

# Save
combined_df.to_csv('combined_dataset.csv', index=False)
print(f"\nSaved to: combined_dataset.csv")

# Show examples from each source
print("\n" + "="*60)
print("SAMPLE REVIEWS:")
print("="*60)

print("\nIMDB Positive:")
print(combined_df[(combined_df['source']=='imdb') & (combined_df['label']==1)].iloc[0]['cleaned_text'][:200])

print("\nYelp Positive:")
print(combined_df[(combined_df['source']=='yelp') & (combined_df['label']==1)].iloc[0]['cleaned_text'][:200])

print("\nIMDB Negative:")
print(combined_df[(combined_df['source']=='imdb') & (combined_df['label']==0)].iloc[0]['cleaned_text'][:200])

print("\nYelp Negative:")
print(combined_df[(combined_df['source']=='yelp') & (combined_df['label']==0)].iloc[0]['cleaned_text'][:200])