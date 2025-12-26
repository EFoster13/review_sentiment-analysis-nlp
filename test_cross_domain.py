import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your trained model (updated path)
print("Loading your multi-domain model...")
model = AutoModelForSequenceClassification.from_pretrained('./models/final_model')
tokenizer = AutoTokenizer.from_pretrained('./models/final_model')

def predict_sentiment(text):
    """Predict sentiment with confidence"""
    text = text.lower()
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction].item()
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return sentiment, confidence

# Test reviews from different domains
print("\n" + "="*60)
print("CROSS-DOMAIN TESTING")
print("="*60)

test_cases = [
    # Movie reviews (IMDB domain)
    ("MOVIE", "This film was a masterpiece! The acting was brilliant and the plot kept me engaged throughout.", "POSITIVE"),
    ("MOVIE", "Terrible movie. Waste of two hours. The plot made no sense and the acting was awful.", "NEGATIVE"),
    
    # Restaurant reviews (Yelp domain)
    ("RESTAURANT", "Amazing food and excellent service! Will definitely come back.", "POSITIVE"),
    ("RESTAURANT", "Horrible experience. Food was cold and the waiter was rude.", "NEGATIVE"),
    
    # Product reviews (new domain - generalization test!)
    ("PRODUCT", "This phone is incredible! Fast, great camera, long battery life.", "POSITIVE"),
    ("PRODUCT", "Broke after one week. Terrible quality. Don't waste your money.", "NEGATIVE"),
    
    # Ambiguous/tricky cases
    ("MIXED", "The movie had some good moments but overall I was disappointed.", "NEGATIVE"),
    ("MIXED", "Not the best restaurant but the dessert was amazing!", "POSITIVE"),
]

correct = 0
total = len(test_cases)

for domain, review, expected in test_cases:
    sentiment, confidence = predict_sentiment(review)
    is_correct = sentiment == expected
    correct += is_correct
    
    status = "CORRECT" if is_correct else "WRONG"
    print(f"\n{status} [{domain}]")
    print(f"Review: {review}")
    print(f"Predicted: {sentiment} ({confidence*100:.1f}%)")
    print(f"Expected: {expected}")
    print("-" * 60)

accuracy = (correct / total) * 100
print(f"\n{'='*60}")
print(f"ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
print(f"{'='*60}")

# Interactive testing
print("\n" + "="*60)
print("TRY YOUR OWN REVIEWS!")
print("="*60)
print("Enter any review (movie, restaurant, product, etc.)")
print("Type 'quit' to exit\n")

while True:
    user_review = input("Your review: ")
    
    if user_review.lower() == 'quit':
        print("\nThanks for testing!")
        break
    
    if user_review.strip() == "":
        continue
    
    sentiment, confidence = predict_sentiment(user_review)
    print(f"→ {sentiment} ({confidence*100:.1f}% confident)\n")