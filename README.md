# sentiment-analysis-amazon
Amazon Review Sentiment Analysis
 ML-powered sentiment classifier achieving 90% accuracy on Amazon product reviews
 
Overview: 
Automatically classifies Amazon product reviews as Positive or Negative using machine learning. Includes an interactive web dashboard for real-time predictions.

Key Results:
. 90% accuracy with Logistic Regression
. Trained on 20,000 reviews
. Processes reviews in <100ms

 Quick Start
bash# Clone repository
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Train models (15-20 minutes)
python train.py

# Launch web app
streamlit run app.py
Open browser at http://localhost:8501

📁 Project Structure
├── data/                   # Dataset files
├── models/                 # Trained models (.pkl files)
├── outputs/               # Visualizations & results
├── src/
│   └── utils.py           # Helper functions
├── train.py               # Model training script
├── app.py                 # Streamlit web app
└── requirements.txt       # Dependencies

Models:
Model                            Accuracy       Speed
Logistic Regression               90%          ⚡⚡⚡
SVM                               89%          ⚡⚡
Random Forest                     88%          ⚡
Naive Bayes                       87%          ⚡⚡⚡

Usage:
Single Prediction
pythonimport joblib
from src.utils import TextPreprocessor

model = joblib.load('models/logistic_regression.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
preprocessor = TextPreprocessor()

review = "This product is amazing!"
cleaned = preprocessor.preprocess(review)
features = vectorizer.transform([cleaned])
prediction = model.predict(features)[0]

print("Positive" if prediction == 1 else "Negative")
Batch Prediction
Upload CSV with review_text column via web dashboard.
 Results :

Accuracy: 90%
Precision: 0.91
Recall: 0.90
F1-Score: 0.90
ROC-AUC: 0.95

Project Statistics

Lines of Code: ~2,000
Training Data: 20,000 reviews
Models Trained: 4 ML algorithms
Best Accuracy: 90%
Training Time: 15-20 minutes
Prediction Speed: <100ms per review

Technologies:

ML: Scikit-learn, NLTK
Web: Streamlit
Data: Pandas, NumPy
Viz: Matplotlib, Seaborn

 Future Work:

 BERT implementation (93-95% accuracy)
 Aspect-based sentiment analysis
 Multilingual support
 Cloud deployment

 License:
MIT License
👤 Author
[renuka dandi]

GitHub: renuka-dandi-ai
LinkedIn: www.linkedin.com/in/renuka-dandi-ai-datascience
