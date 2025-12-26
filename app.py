import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page config
st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="🎭",
    layout="centered"
)

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('./models/final_model')
    tokenizer = AutoTokenizer.from_pretrained('./models/final_model')
    return model, tokenizer

model, tokenizer = load_model()

# Prediction function
def predict_sentiment(text):
    text = text.lower()
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

# App UI
st.title("Multi-Domain Review Sentiment Analyzer")
st.markdown("Analyze sentiment across reviews for movies, restaurants, products, and more!")

# Text input
user_input = st.text_area(
    "Enter your review:",
    placeholder="Type a review about a movie, restaurant, product, etc...",
    height=150
)

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            prediction, confidence = predict_sentiment(user_input)
            
            sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
            
            # Display result
            if sentiment == "POSITIVE":
                st.success(f"### 😊 {sentiment}")
            else:
                st.error(f"### 😞 {sentiment}")
            
            st.metric("Confidence", f"{confidence*100:.1f}%")
    else:
        st.warning("Please enter a review first!")

# Examples
with st.expander("See Example Reviews"):
    st.markdown("""
    **Positive Examples:**
    - "This movie was amazing! The acting was brilliant."
    - "Best restaurant I've ever been to. Food was incredible."
    - "Love this product! Works perfectly and great quality."
    
    **Negative Examples:**
    - "Terrible movie. Complete waste of time."
    - "Worst dining experience ever. Food was cold."
    - "Product broke after one day. Very disappointed."
    """)

# Footer
st.markdown("---")
st.markdown("Built with BERT transformer + Trained on 30K+ reviews")