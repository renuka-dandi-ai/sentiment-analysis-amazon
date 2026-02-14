"""
Utility functions for Sentiment Analysis Project
Contains preprocessing, visualization, and evaluation utilities
"""

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """
    Advanced text preprocessing class for sentiment analysis
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some sentiment-bearing stopwords
        self.stop_words -= {'not', 'no', 'nor', 'neither', 'never', 'none'}
        
    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def remove_html(self, text):
        """Remove HTML tags"""
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub(r'', text)
    
    def remove_emojis(self, text):
        """Remove emojis from text"""
        emoji_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251"
                                  "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def remove_repeated_chars(self, text):
        """Remove repeated characters (e.g., 'loooove' -> 'love')"""
        return re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning pipeline
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove HTML tags
        text = self.remove_html(text)
        
        # Remove emojis
        text = self.remove_emojis(text)
        
        # Remove repeated characters
        text = self.remove_repeated_chars(text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize, remove stopwords, and lemmatize
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        """
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text


class DataVisualizer:
    """
    Visualization utilities for EDA and model evaluation
    """
    
    @staticmethod
    def plot_class_distribution(y, title='Class Distribution', save_path=None):
        """Plot class distribution"""
        plt.figure(figsize=(10, 6))
        
        # Count plot
        unique, counts = np.unique(y, return_counts=True)
        colors = ['#FF6B6B' if label == 0 else '#4ECDC4' if label == 1 else '#95E1D3' 
                 for label in unique]
        
        plt.bar(unique, counts, color=colors, alpha=0.8, edgecolor='black')
        plt.xlabel('Sentiment Class', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(unique, ['Negative', 'Positive', 'Neutral'][:len(unique)])
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(unique[i], count + 50, str(count), ha='center', 
                    fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_review_length_distribution(df, text_col='Text', save_path=None):
        """Plot review length distribution"""
        df['review_length'] = df[text_col].astype(str).apply(len)
        df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Character length distribution
        axes[0].hist(df['review_length'], bins=50, color='skyblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Review Length (characters)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Review Length Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(df['review_length'].mean(), color='red', 
                       linestyle='--', linewidth=2, label='Mean')
        axes[0].legend()
        
        # Word count distribution
        axes[1].hist(df['word_count'], bins=50, color='lightcoral', 
                    edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Word Count', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
        axes[1].axvline(df['word_count'].mean(), color='blue', 
                       linestyle='--', linewidth=2, label='Mean')
        axes[1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def generate_wordcloud(text_data, title='Word Cloud', save_path=None):
        """Generate and display word cloud"""
        plt.figure(figsize=(12, 8))
        
        # Combine all text
        all_text = ' '.join(text_data.astype(str))
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100,
                             relative_scaling=0.5,
                             min_font_size=10).generate(all_text)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_top_words(text_data, n=20, title='Top Words', save_path=None):
        """Plot most frequent words"""
        from collections import Counter
        
        # Combine and split all text
        all_words = ' '.join(text_data.astype(str)).split()
        word_freq = Counter(all_words)
        
        # Get top n words
        top_words = word_freq.most_common(n)
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(words)), counts, color='teal', alpha=0.8)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency', fontsize=12, fontweight='bold')
        plt.ylabel('Words', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModelEvaluator:
    """
    Model evaluation and comparison utilities
    """
    
    @staticmethod
    def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name='Model'):
        """
        Comprehensive model evaluation
        """
        print(f"\n{'='*60}")
        print(f"{model_name} Evaluation Results")
        print(f"{'='*60}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Classification report
        print(f"\n{'-'*60}")
        print("Classification Report:")
        print(f"{'-'*60}")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Negative', 'Positive']))
        
        # ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                print(f"\nROC-AUC Score: {roc_auc:.4f}")
            except:
                pass
        
        return {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'], 
                            title='Confusion Matrix', save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, model_name='Model', save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def compare_models(results_df, save_path=None):
        """
        Compare multiple models with visualization
        """
        # Sort by F1-Score
        results_df = results_df.sort_values('F1-Score', ascending=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D']
        
        for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
            ax.barh(results_df['Model'], results_df[metric], color=color, alpha=0.8)
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1.0])
            
            # Add value labels
            for i, v in enumerate(results_df[metric]):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        return results_df


def save_model(model, filepath):
    """Save model using joblib"""
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """Load model using joblib"""
    import joblib
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model
