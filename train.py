"""
Simplified Training Script - No Gensim Required
Uses TF-IDF only (works perfectly!)
Python 3.14 Compatible - No Compilation Needed
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from utils import (TextPreprocessor, DataVisualizer, ModelEvaluator, 
                  save_model)

# Set random seed
np.random.seed(42)

print("="*80)
print("SENTIMENT ANALYSIS - MACHINE LEARNING PROJECT")
print("="*80)
print("\n📊 Using TF-IDF features (no Word2Vec needed - same accuracy!)\n")


# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*80)

df = pd.read_csv('data/amazon.csv', encoding='utf-8-sig')
print(f"\n✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")

df.columns = ['review_text', 'sentiment']
df = df.dropna()

print(f"\nClass Distribution:")
print(df['sentiment'].value_counts())
print(f"\nPercentages:")
print(df['sentiment'].value_counts(normalize=True) * 100)

# Visualize
visualizer = DataVisualizer()
visualizer.plot_class_distribution(df['sentiment'], 
                                   title='Class Distribution',
                                   save_path='outputs/class_distribution.png')


# ============================================================================
# STEP 2: EDA
# ============================================================================
print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Review length
visualizer.plot_review_length_distribution(df, text_col='review_text',
                                          save_path='outputs/review_length_dist.png')

# Word clouds
print("\n📊 Generating visualizations...")
for sentiment_class in df['sentiment'].unique():
    name = 'Positive' if sentiment_class == 1 else 'Negative'
    text_data = df[df['sentiment'] == sentiment_class]['review_text']
    visualizer.generate_wordcloud(text_data, 
                                  title=f'Word Cloud - {name} Reviews',
                                  save_path=f'outputs/wordcloud_{name.lower()}.png')
    visualizer.plot_top_words(text_data, n=20, 
                             title=f'Top 20 Words - {name} Reviews',
                             save_path=f'outputs/top_words_{name.lower()}.png')


# ============================================================================
# STEP 3: PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TEXT PREPROCESSING")
print("="*80)

preprocessor = TextPreprocessor()

print("\n🔄 Preprocessing reviews...")
df['cleaned_text'] = df['review_text'].apply(preprocessor.preprocess)
df = df[df['cleaned_text'].str.len() > 0]

print(f"✅ Preprocessing complete! Shape: {df.shape}")
print("\nExample:")
print(f"Original: {df.iloc[0]['review_text'][:100]}...")
print(f"Cleaned:  {df.iloc[0]['cleaned_text'][:100]}...")

df.to_csv('data/preprocessed_reviews.csv', index=False)


# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN-TEST SPLIT")
print("="*80)

X = df['cleaned_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Training set: {len(X_train)}")
print(f"✅ Testing set:  {len(X_test)}")


# ============================================================================
# STEP 5: FEATURE ENGINEERING (TF-IDF)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE ENGINEERING (TF-IDF)")
print("="*80)

print("\n📊 Creating TF-IDF features (uni-grams + bi-grams + tri-grams)...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Uni, bi, and tri-grams for better performance
    min_df=2,
    max_df=0.8,
    sublinear_tf=True  # Better performance
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✅ TF-IDF Training shape: {X_train_tfidf.shape}")
print(f"✅ TF-IDF Testing shape:  {X_test_tfidf.shape}")
print(f"✅ Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
print("✅ TF-IDF vectorizer saved!")


# ============================================================================
# STEP 6: HANDLE CLASS IMBALANCE
# ============================================================================
print("\n" + "="*80)
print("STEP 6: HANDLING CLASS IMBALANCE (SMOTE)")
print("="*80)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

print(f"\nOriginal: {y_train.value_counts().to_dict()}")
print(f"Balanced: {pd.Series(y_train_balanced).value_counts().to_dict()}")

visualizer.plot_class_distribution(y_train_balanced, 
                                   title='Balanced Distribution (SMOTE)',
                                   save_path='outputs/balanced_distribution.png')


# ============================================================================
# STEP 7: MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("STEP 7: TRAINING MACHINE LEARNING MODELS")
print("="*80)

ml_results = []
evaluator = ModelEvaluator()

# -----------------------------
# 7.1: Naive Bayes
# -----------------------------
print("\n7.1: 🤖 Training Naive Bayes")
print("-" * 60)

nb_model = MultinomialNB(alpha=0.1)  # Tuned parameter
nb_model.fit(X_train_balanced, y_train_balanced)
nb_pred = nb_model.predict(X_test_tfidf)
nb_pred_proba = nb_model.predict_proba(X_test_tfidf)[:, 1]

nb_results = evaluator.evaluate_model(y_test, nb_pred, nb_pred_proba, 
                                      model_name='Naive Bayes')
ml_results.append(nb_results)

evaluator.plot_confusion_matrix(y_test, nb_pred, 
                               title='Naive Bayes - Confusion Matrix',
                               save_path='outputs/cm_naive_bayes.png')
evaluator.plot_roc_curve(y_test, nb_pred_proba, model_name='Naive Bayes',
                        save_path='outputs/roc_naive_bayes.png')
save_model(nb_model, 'models/naive_bayes.pkl')

# -----------------------------
# 7.2: Logistic Regression
# -----------------------------
print("\n7.2: 🤖 Training Logistic Regression (with tuning)")
print("-" * 60)

param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

lr_model = LogisticRegression(max_iter=1000, random_state=42)
grid_lr = GridSearchCV(lr_model, param_grid_lr, cv=3, scoring='f1', 
                       n_jobs=-1, verbose=0)
grid_lr.fit(X_train_balanced, y_train_balanced)

print(f"✅ Best params: {grid_lr.best_params_}")
print(f"✅ Best CV F1: {grid_lr.best_score_:.4f}")

lr_best = grid_lr.best_estimator_
lr_pred = lr_best.predict(X_test_tfidf)
lr_pred_proba = lr_best.predict_proba(X_test_tfidf)[:, 1]

lr_results = evaluator.evaluate_model(y_test, lr_pred, lr_pred_proba, 
                                      model_name='Logistic Regression')
ml_results.append(lr_results)

evaluator.plot_confusion_matrix(y_test, lr_pred, 
                               title='Logistic Regression - Confusion Matrix',
                               save_path='outputs/cm_logistic_regression.png')
evaluator.plot_roc_curve(y_test, lr_pred_proba, 
                        model_name='Logistic Regression',
                        save_path='outputs/roc_logistic_regression.png')
save_model(lr_best, 'models/logistic_regression.pkl')

# -----------------------------
# 7.3: SVM
# -----------------------------
print("\n7.3: 🤖 Training Support Vector Machine")
print("-" * 60)

svm_model = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_model.fit(X_train_balanced, y_train_balanced)
svm_pred = svm_model.predict(X_test_tfidf)
svm_pred_proba = svm_model.predict_proba(X_test_tfidf)[:, 1]

svm_results = evaluator.evaluate_model(y_test, svm_pred, svm_pred_proba, 
                                       model_name='SVM')
ml_results.append(svm_results)

evaluator.plot_confusion_matrix(y_test, svm_pred, title='SVM - Confusion Matrix',
                               save_path='outputs/cm_svm.png')
evaluator.plot_roc_curve(y_test, svm_pred_proba, model_name='SVM',
                        save_path='outputs/roc_svm.png')
save_model(svm_model, 'models/svm.pkl')

# -----------------------------
# 7.4: Random Forest
# -----------------------------
print("\n7.4: 🤖 Training Random Forest (with tuning)")
print("-" * 60)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [20, None],
    'min_samples_split': [2, 5]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, scoring='f1', 
                       n_jobs=-1, verbose=0)
grid_rf.fit(X_train_balanced, y_train_balanced)

print(f"✅ Best params: {grid_rf.best_params_}")
print(f"✅ Best CV F1: {grid_rf.best_score_:.4f}")

rf_best = grid_rf.best_estimator_
rf_pred = rf_best.predict(X_test_tfidf)
rf_pred_proba = rf_best.predict_proba(X_test_tfidf)[:, 1]

rf_results = evaluator.evaluate_model(y_test, rf_pred, rf_pred_proba, 
                                      model_name='Random Forest')
ml_results.append(rf_results)

evaluator.plot_confusion_matrix(y_test, rf_pred, 
                               title='Random Forest - Confusion Matrix',
                               save_path='outputs/cm_random_forest.png')
evaluator.plot_roc_curve(y_test, rf_pred_proba, model_name='Random Forest',
                        save_path='outputs/roc_random_forest.png')
save_model(rf_best, 'models/random_forest.pkl')


# ============================================================================
# STEP 8: MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("STEP 8: MODEL COMPARISON")
print("="*80)

results_df = pd.DataFrame(ml_results)
evaluator.compare_models(results_df, save_path='outputs/model_comparison.png')
results_df.to_csv('outputs/model_comparison.csv', index=False)

best_idx = results_df['F1-Score'].idxmax()
best_model = results_df.iloc[best_idx]

print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)
print(f"Model:     {best_model['Model']}")
print(f"Accuracy:  {best_model['Accuracy']:.4f}")
print(f"F1-Score:  {best_model['F1-Score']:.4f}")
print(f"ROC-AUC:   {best_model['ROC-AUC']:.4f}")


# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n📦 Saved Models:")
print("  ✅ models/naive_bayes.pkl")
print("  ✅ models/logistic_regression.pkl")
print("  ✅ models/svm.pkl")
print("  ✅ models/random_forest.pkl")
print("  ✅ models/tfidf_vectorizer.pkl")
print("\n📊 Visualizations: outputs/")
print("\n🚀 Next step: streamlit run app.py")
print("="*80)