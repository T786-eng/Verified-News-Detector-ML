import streamlit as st
import joblib
import re
import string

# 1. Load the saved models
@st.cache_resource # Use caching to load models only once
def load_assets():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model

vectorizer, model = load_assets()

# 2. Text Cleaning Function (MUST match your main.py cleaning logic)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\W', ' ', text)
    return text.strip()

# 3. Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.markdown("""
This app uses a **Machine Learning model** to predict if a news article is likely 
**Real** or **Fake** based on its text content.
""")

input_text = st.text_area("Paste the News Article below:", height=200)

if st.button("Analyze News"):
    if input_text.strip():
        # Step A: Clean the user input
        cleaned_input = clean_text(input_text)
        
        # Step B: Transform using the loaded vectorizer
        transform_input = vectorizer.transform([cleaned_input])
        
        # Step C: Predict
        prediction = model.predict(transform_input)
        
        # Step D: Display Results
        st.divider()
        if prediction[0] == 1:
            st.success("### ✅ Result: The News is likely REAL.")
        else:
            st.error("### ⚠️ Result: The News is likely FAKE.")
            
        # Optional: Show confidence/probability
        prob = model.predict_proba(transform_input)
        st.write(f"Model Confidence: **{max(prob[0]) * 100:.2f}%**")
        
    else:
        st.warning("Please enter some text to analyze.")