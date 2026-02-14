"""
Streamlit Web Application for Sentiment Analysis (ML Models Only)
Works without TensorFlow - Python 3.14 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils import TextPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 2rem;
    }
    .positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .negative {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD MODELS AND ARTIFACTS
# =============================================================================
@st.cache_resource
def load_models_and_artifacts():
    """Load all trained ML models and preprocessing artifacts"""
    
    models = {}
    
    # Check if models directory exists
    if not os.path.exists('models'):
        st.error("Models directory not found! Please train the models first by running train_ml_only.py")
        return None
    
    try:
        # Load ML models
        models['Naive Bayes'] = joblib.load('models/naive_bayes.pkl')
        models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
        models['SVM'] = joblib.load('models/svm.pkl')
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
        
        # Load artifacts
        models['tfidf'] = joblib.load('models/tfidf_vectorizer.pkl')
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure you have trained the models by running: python train_ml_only.py")
        return None


@st.cache_resource
def load_preprocessor():
    """Load text preprocessor"""
    return TextPreprocessor()


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def predict_sentiment_ml(text, model, tfidf_vectorizer, preprocessor):
    """Predict sentiment using ML models"""
    # Preprocess
    cleaned_text = preprocessor.preprocess(text)
    
    # Vectorize
    text_vector = tfidf_vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return prediction, probability


def get_sentiment_label(prediction):
    """Convert prediction to sentiment label"""
    return "Positive 😊" if prediction == 1 else "Negative 😞"


def get_sentiment_color(prediction):
    """Get color for sentiment"""
    return "#2ecc71" if prediction == 1 else "#e74c3c"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_probability_gauge(probability, prediction):
    """Create a gauge chart for probability"""
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction] * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{sentiment} Confidence", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "#2ecc71"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_sentiment_color(prediction)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccb'},
                {'range': [50, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_probability_bars(probability):
    """Create bar chart for probabilities"""
    labels = ['Negative', 'Positive']
    colors = ['#e74c3c', '#2ecc71']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probability * 100,
            marker_color=colors,
            text=[f'{p:.2f}%' for p in probability * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Sentiment Probability Distribution',
        xaxis_title='Sentiment',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )
    
    return fig


def plot_batch_results(df):
    """Plot batch prediction results"""
    sentiment_counts = df['Predicted Sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        color_discrete_sequence=['#e74c3c', '#2ecc71'],
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Analyze customer reviews using Machine Learning")
    
    # Load models
    with st.spinner("Loading models and artifacts..."):
        models = load_models_and_artifacts()
        preprocessor = load_preprocessor()
    
    if models is None:
        st.stop()
    
    st.success("✅ All ML models loaded successfully!")
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", 
                     width=100)
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Machine Learning Model",
        ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest"]
    )
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["Single Prediction", "Batch Prediction"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About**
    
    This dashboard uses machine learning models 
    to analyze sentiment in product reviews.
    
    **Models Available:**
    - Naive Bayes (Fast & Efficient)
    - Logistic Regression (Balanced Performance)
    - Support Vector Machine (High Accuracy)
    - Random Forest (Ensemble Method)
    
    **Note:** Deep learning models require Python 3.11 or lower.
    These ML models often perform just as well!
    """)
    
    # Main content
    if mode == "Single Prediction":
        st.markdown('<h2 class="sub-header">🔍 Single Review Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Text input
        user_input = st.text_area(
            "Enter a product review:",
            height=150,
            placeholder="Type or paste a review here..."
        )
        
        # Predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🎯 Analyze Sentiment", type="primary", 
                                      use_container_width=True)
        
        if predict_button and user_input:
            with st.spinner("Analyzing sentiment..."):
                
                # Make prediction
                prediction, probability = predict_sentiment_ml(
                    user_input, 
                    models[selected_model],
                    models['tfidf'],
                    preprocessor
                )
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Sentiment result
                sentiment_label = get_sentiment_label(prediction)
                sentiment_color = get_sentiment_color(prediction)
                
                st.markdown(
                    f"<div style='text-align: center; padding: 2rem; "
                    f"background-color: {sentiment_color}22; border-radius: 10px; "
                    f"border: 2px solid {sentiment_color};'>"
                    f"<h1 style='color: {sentiment_color}; margin: 0;'>{sentiment_label}</h1>"
                    f"<p style='font-size: 1.2rem; margin-top: 0.5rem;'>"
                    f"Confidence: {probability[prediction]*100:.2f}%</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plot_probability_gauge(probability, prediction), 
                                   use_container_width=True)
                
                with col2:
                    st.plotly_chart(plot_probability_bars(probability), 
                                   use_container_width=True)
                
                # Detailed probabilities
                st.markdown("---")
                st.markdown("### 📈 Detailed Probabilities")
                
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Positive'],
                    'Probability': probability,
                    'Percentage': [f'{p*100:.2f}%' for p in probability]
                })
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
        elif predict_button:
            st.warning("⚠️ Please enter a review to analyze!")
    
    else:  # Batch Prediction
        st.markdown('<h2 class="sub-header">📁 Batch Review Analysis</h2>', 
                   unsafe_allow_html=True)
        
        st.info("Upload a CSV file with a column named 'review_text' containing reviews")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Found {len(df)} reviews")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check for required column
                if 'review_text' not in df.columns and 'Text' not in df.columns:
                    st.error("❌ CSV must contain 'review_text' or 'Text' column!")
                    st.stop()
                
                # Rename column if needed
                if 'Text' in df.columns:
                    df.rename(columns={'Text': 'review_text'}, inplace=True)
                
                # Predict button
                if st.button("🎯 Analyze All Reviews", type="primary"):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = []
                    probabilities = []
                    
                    for idx, review in enumerate(df['review_text']):
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing review {idx+1}/{len(df)}...")
                        
                        # Make prediction
                        pred, prob = predict_sentiment_ml(
                            review, 
                            models[selected_model],
                            models['tfidf'],
                            preprocessor
                        )
                        
                        predictions.append(pred)
                        probabilities.append(prob[pred])
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Add results to dataframe
                    df['Predicted Sentiment'] = ['Positive' if p == 1 else 'Negative' 
                                                 for p in predictions]
                    df['Confidence'] = [f'{p*100:.2f}%' for p in probabilities]
                    
                    st.success("✅ Analysis complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total = len(df)
                    positive = sum(predictions)
                    negative = total - positive
                    avg_confidence = np.mean(probabilities) * 100
                    
                    with col1:
                        st.metric("Total Reviews", total)
                    with col2:
                        st.metric("Positive", positive, 
                                 delta=f"{positive/total*100:.1f}%")
                    with col3:
                        st.metric("Negative", negative, 
                                 delta=f"{negative/total*100:.1f}%", 
                                 delta_color="inverse")
                    with col4:
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                    
                    # Visualization
                    st.plotly_chart(plot_batch_results(df), use_container_width=True)
                    
                    # Results table
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")


# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()