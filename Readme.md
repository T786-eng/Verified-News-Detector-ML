# 📰 Verified-News-Detector-ML

A Machine Learning-powered application designed to verify the authenticity of news articles using Natural Language Processing (NLP). This tool classifies text as **Real** or **Fake** by analyzing linguistic patterns and metadata.

---

## 🚀 Project Overview
This project focuses on the critical problem of misinformation in the digital age. By leveraging supervised learning techniques, the model provides a reliable way to assess news content. It features a complete pipeline from raw data preprocessing to a live interactive dashboard.

## 🛠️ Key Features
- **Advanced Text Cleaning**: A custom pipeline to handle noisy data by removing URLs, HTML tags, punctuation, and digits.
- **Robust Classification**: Uses a Logistic Regression model trained for high-precision text classification.
- **Feature Engineering**: Implements TF-IDF Vectorization to convert textual data into meaningful numerical features.
- **Live Dashboard**: An intuitive user interface built with **Streamlit** for real-time predictions.

## 📂 Project Structure
- `main.py`: Core script for data cleaning, training, and model evaluation.
- `app.py`: Streamlit application script for the web UI.
- `lr_model.jb`: Optimized Logistic Regression model file.
- `vectorizer.jb`: Trained TF-IDF vectorizer object.
- `requirements.txt`: Project dependencies and library versions.

## 🔧 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/T786-eng/Verified-News-Detector-ML.git](https://github.com/T786-eng/Verified-News-Detector-ML.git)
   cd Verified-News-Detector-ML


2. **Set up Environment:**
    ```bash
    # Create a virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

3. **install Dependencies:**
    ```bash
    pip install -r requirements.txt
     ```

💻 Usage
Run the following command to launch the local web server:
    ```bash
    streamlit run app.py
    ```

📊 Technical Stack
- Language: Python

- Libraries: Scikit-learn, Pandas, Joblib, Streamlit, NLTK

- Vectorization: TF-IDF

- Model: Logistic Regression