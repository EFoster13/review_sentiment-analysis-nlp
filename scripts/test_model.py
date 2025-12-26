import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
print("Loading the trained model...")
model = AutoModelForSequenceClassification.from_pretrained('./results/checkpoint-200')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def predict_sentiment(review_text):
    """
    Predict sentiment of a movie review
    
    Returns: 'POSITIVE' or 'NEGATIVE' with confidence score
    """
    # Clean the text (same as training)
    review_text = review_text.lower()
    
    # Tokenize
    inputs = tokenizer(review_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():  # Don't calculate gradients (faster)
        outputs = model(**inputs)
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction].item()
    
    # Convert to label
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
    return sentiment, confidence

# Test reviews
print("\n" + "="*60)
print("TESTING THE MODEL")
print("="*60)

test_reviews = [
    "This movie was absolutely amazing! Best film I've seen all year.",
    "Terrible waste of time. The acting was horrible and the plot made no sense.",
    "It was okay, nothing special but not bad either.",
    "I loved every minute of this masterpiece. Brilliant directing and acting!",
    "Worst movie ever. I walked out halfway through.",
]

for i, review in enumerate(test_reviews, 1):
    sentiment, confidence = predict_sentiment(review)
    
    print(f"\nReview {i}:")
    print(f"Text: {review}")
    print(f"Prediction: {sentiment} ({confidence*100:.1f}% confident)")
    print("-" * 60)

# Interactive mode - test your own reviews
print("\n" + "="*60)
print("TRY A CUSTOM REVIEW")
print("="*60)
print("Enter a movie review (or 'quit' to exit):\n")

while True:
    user_review = input("Your review: ")
    
    if user_review.lower() == 'quit':
        print("\nThanks for testing!")
        break
    
    if user_review.strip() == "":
        continue
    
    sentiment, confidence = predict_sentiment(user_review)
    print(f"→ {sentiment} ({confidence*100:.1f}% confident)\n")