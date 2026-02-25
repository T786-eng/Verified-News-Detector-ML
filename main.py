import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def clean_text(text):
    """Standardizes text by removing punctuation, URLs, and numbers."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    # Fixed URL regex
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove non-word characters and extra whitespace
    text = re.sub(r'\W', ' ', text)
    return text.strip()

def run_pipeline():
    try:
        # Load datasets
        print("Loading data...")
        fake = pd.read_csv('Fake.csv')
        true = pd.read_csv('True.csv')

        # Labeling
        fake['class'] = 0
        true['class'] = 1

        # Combine and Shuffle to prevent ordering bias
        data = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)

        # Keep only necessary columns to save memory
        data = data[['text', 'class']]

        print("Cleaning text (this may take a moment)...")
        data['text'] = data['text'].apply(clean_text)

        x = data["text"]
        y = data["class"]

        # Split data
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)

        # Vectorization
        print("Vectorizing data...")
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        xv_train = vectorizer.fit_transform(xtrain)
        xv_test = vectorizer.transform(xtest)

        # Model Training
        print("Training Logistic Regression model...")
        lr = LogisticRegression(max_iter=1000)
        lr.fit(xv_train, ytrain)

        # Evaluation
        prediction = lr.predict(xv_test)
        print("\nModel Accuracy:", lr.score(xv_test, ytest))
        print("\nClassification Report:\n")
        print(classification_report(ytest, prediction))

        # Save models
        print("Saving models...")
        joblib.dump(vectorizer, "vectorizer.jb")
        joblib.dump(lr, "lr_model.jb")
        print("Success! 'vectorizer.jb' and 'lr_model.jb' are ready.")

    except FileNotFoundError:
        print("Error: 'Fake.csv' or 'True.csv' not found. Please ensure they are in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_pipeline()