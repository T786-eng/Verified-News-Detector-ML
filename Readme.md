# 📰 Verified-News-Detector-ML
> **Supervised NLP Intelligence for Real-Time Misinformation Classification**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://verified-news-detector-ml-uqm7hn4ptutgo4xvhvivkd.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Standard: Clean Code](https://img.shields.io/badge/Standard-Clean--Code-brightgreen.svg)](#)

Verified-News-Detector-ML is a high-precision analytics tool designed to verify the authenticity of news articles. By leveraging **Logistic Regression** and **TF-IDF Vectorization**, the system identifies deceptive linguistic patterns to combat digital misinformation.

---

## 🚀 Live Dashboard
The system is fully deployed and accessible here:  
🔗 **[Verified News Detector Live Interface](https://verified-news-detector-ml-uqm7hn4ptutgo4xvhvivkd.streamlit.app/)**

---

## 🏗️ System Architecture
The application follows a modular architecture designed for high-speed text inference:
1. **Preprocessing Engine**: Custom regex-based pipeline that standardizes raw text by removing URLs, HTML tags, and punctuation.
2. **Feature Engineering Layer**: Implements **TF-IDF Vectorization** to convert textual content into high-dimensional numerical features.
3. **Inference Layer**: A serialized **Logistic Regression** model providing sub-second classification between "Real" and "Fake" categories.

---

## ✨ Engineering Highlights
* ✅ **Production Ready**: Fully refactored for 2026 Streamlit standards with zero deprecation warnings.
* ✅ **Efficient Serialization**: Utilizes `joblib` for model persistence, ensuring rapid application startup.
* ✅ **Advanced Data Cleaning**: Implements a robust cleaning script that removes non-word characters and numeric noise without losing semantic integrity.
* ✅ **Scalable Design**: Modular code structure allowing for easy integration of more complex models like LSTMs or Transformers.

---

## 📂 Project Structure
```text
Verified-News-Detector-ML/
│
├── main.py                      # Model Training & Preprocessing Engine
├── app.py                       # Modular Streamlit Dashboard
├── requirements.txt             # Dependency Manifest
├── README.md                    # System Documentation
│
├── Models/                      # Serialized ML Artifacts
│   ├── lr_model.jb              # Optimized Logistic Regression Model
│   └── vectorizer.jb            # Trained TF-IDF Vectorizer Object
│
└── Data/                        # Input Datasets
    ├── Fake.csv                 # Misinformation Training Data
    └── True.csv                 # Verified News Training Data

```

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
```
    .\venv\Scripts\activate
```

# Activate (macOS/Linux)
```
    source venv/bin/activate
```

3. **install Dependencies:**
    ```bash
    pip install -r requirements.txt
     ```

💻 Usage
Run the following command to launch the local web server:
```
    streamlit run app.py
```


📊 Technical Stack
- Language: Python

- Libraries: Scikit-learn, Pandas, Joblib, Streamlit, NLTK

- Vectorization: TF-IDF

- Model: Logistic Regression


