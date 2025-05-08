import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModel
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Debug: Print versions
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)  # Works in 2.12.0
print("Streamlit version:", st.__version__)

# Fix Functional deserialization
tf.keras.utils.get_custom_objects().update({'Functional': tf.keras.models.Model})

# Custom Attention Layer
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ScaledDotProductAttention, self).build(input_shape)
    def call(self, inputs):
        query, key, value = inputs
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(d_k)
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, value)
        return output

# Fix InputLayer deserialization
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super(CustomInputLayer, self).__init__(**kwargs)

# Define a more compatible DTypePolicy with get_config()
class DTypePolicy:
    def __init__(self, name=None, **kwargs):
        self.name = name if name else 'float32'
        self.compute_dtype = self.name
        self.variable_dtype = self.name
    def get_config(self):
        return {
            'name': self.name,
            'compute_dtype': self.compute_dtype,
            'variable_dtype': self.variable_dtype
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    @staticmethod
    def deserialize(config):
        return DTypePolicy(**config)

# Register custom objects
tf.keras.utils.get_custom_objects().update({
    'InputLayer': CustomInputLayer,
    'DTypePolicy': DTypePolicy,
    'ScaledDotProductAttention': ScaledDotProductAttention
})

# Set local path to models and data
drive_model_path = './StockPrediction_Model'  # Adjust to 'D:/DLP_Project/StockPrediction_Model' if needed
print(f"Using drive_model_path: {drive_model_path}")
print("Files in directory:", os.listdir(drive_model_path))

# Load FinBERT model to ensure weights are downloaded
print("Pre-loading FinBERT model for custom object registration...")
finbert_model = TFAutoModel.from_pretrained("yiyanghkust/finbert-tone")
print("FinBERT model pre-loaded.")

# Load models with custom object scope
with tf.keras.utils.custom_object_scope({
    'ScaledDotProductAttention': ScaledDotProductAttention,
    'TFBertModel': lambda: finbert_model,
    'InputLayer': CustomInputLayer,
    'DTypePolicy': DTypePolicy
}):
    print("Loading LSTM+FinBERT model...")
    try:
        lstm_model = tf.keras.models.load_model(
            os.path.join(drive_model_path, 'best_model.keras'),
            compile=False
        )
        print("LSTM+FinBERT model loaded successfully.")
    except Exception as e:
        print(f"Error loading LSTM+FinBERT model: {e}")
        raise
    print("Loading RNN model...")
    try:
        rnn_model = tf.keras.models.load_model(
            os.path.join(drive_model_path, 'rnn_model.keras'),
            compile=False
        )
        print("RNN model loaded successfully.")
    except Exception as e:
        print(f"Error loading RNN model: {e}")
        raise

print("Loading Random Forest model...")
rf_model = joblib.load(os.path.join(drive_model_path, 'rf_model.pkl'))
print("Random Forest model loaded.")

# Load data
print("Loading preprocessed_data.csv...")
full_df = pd.read_csv(os.path.join(drive_model_path, 'preprocessed_data.csv'), parse_dates=['Date'])
print("preprocessed_data.csv loaded.")

print("Loading numpy arrays...")
X_test_num = np.load(os.path.join(drive_model_path, 'X_test_num.npy'))
y_test = np.load(os.path.join(drive_model_path, 'y_test.npy'))
test_encodings_input_ids = np.load(os.path.join(drive_model_path, 'test_encodings_input_ids.npy'))
test_encodings_attention_mask = np.load(os.path.join(drive_model_path, 'test_encodings_attention_mask.npy'))
print("Numpy arrays loaded.")

print("Loading model_comparison.csv...")
comparison_df = pd.read_csv(os.path.join(drive_model_path, 'model_comparison.csv'))
print("model_comparison.csv loaded.")

# Streamlit app
st.title("Stock Price Prediction Dashboard")
st.markdown("""
This dashboard predicts DJIA stock price movement (Up/Down) using three models: LSTM+FinBERT, Random Forest, and RNN.
Explore predictions, model performance, comparisons, and data visualizations.
""")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Predictions", "Model Performance", "Model Comparison", "Data Visualization"])

if page == "Home":
    st.header("Project Overview")
    st.markdown("""
    **Goal**: Predict whether the Dow Jones Industrial Average (DJIA) will go up or down based on historical stock data and news sentiment.
    **Dataset**:
    - Training samples: 2798
    - Test samples: 358
    - Features: 25 (stock prices, technical indicators, FinBERT sentiment scores)
    - Window size: 20 days
    **Models**:
    - **LSTM+FinBERT**: Combines LSTM for sequential numerical data and FinBERT for news sentiment.
    - **Random Forest**: Non-sequential classifier using flattened features.
    - **RNN**: Simple RNN with FinBERT for sequential data.
    """)

elif page == "Predictions":
    st.header("Stock Price Predictions")
    test_dates = full_df[full_df['Date'] >= '2015-01-01']['Date'].iloc[20:].reset_index(drop=True)
    selected_date = st.selectbox("Select a test set date", test_dates)
    idx = test_dates[test_dates == selected_date].index[0]
    
    X_num = X_test_num[idx:idx+1]
    input_ids = test_encodings_input_ids[idx:idx+1]
    attention_mask = test_encodings_attention_mask[idx:idx+1]
    X_rf = X_num.reshape(1, -1)
    
    pred_lstm = lstm_model.predict({'num_input': X_num, 'input_ids': input_ids, 'attention_mask': attention_mask}, verbose=0).flatten()[0]
    pred_rf = rf_model.predict_proba(X_rf)[:, 1][0]
    pred_rnn = rnn_model.predict({'num_input': X_num, 'input_ids': input_ids, 'attention_mask': attention_mask}, verbose=0).flatten()[0]
    
    st.markdown(f"**Predictions for {selected_date.date()}**:")
    st.write(f"- **LSTM+FinBERT**: {'Up' if pred_lstm > 0.5 else 'Down'} (Probability: {pred_lstm:.2f})")
    st.write(f"- **Random Forest**: {'Up' if pred_rf > 0.5 else 'Down'} (Probability: {pred_rf:.2f})")
    st.write(f"- **RNN**: {'Up' if pred_rnn > 0.5 else 'Down'} (Probability: {pred_rnn:.2f})")

elif page == "Model Performance":
    st.header("Model Performance")
    model_choice = st.selectbox("Select Model", ["LSTM+FinBERT", "Random Forest", "RNN"])
    
    if model_choice == "LSTM+FinBERT":
        y_pred = lstm_model.predict({'num_input': X_test_num, 'input_ids': test_encodings_input_ids, 'attention_mask': test_encodings_attention_mask}, verbose=0).flatten()
        report = classification_report(y_test, (y_pred > 0.5).astype(int), output_dict=True)
        auc = roc_auc_score(y_test, y_pred)
    elif model_choice == "Random Forest":
        y_pred = rf_model.predict_proba(X_test_num.reshape(X_test_num.shape[0], -1))[:, 1]
        report = classification_report(y_test, (y_pred > 0.5).astype(int), output_dict=True)
        auc = roc_auc_score(y_test, y_pred)
    else:
        y_pred = rnn_model.predict({'num_input': X_test_num, 'input_ids': test_encodings_input_ids, 'attention_mask': test_encodings_attention_mask}, verbose=0).flatten()
        report = classification_report(y_test, (y_pred > 0.5).astype(int), output_dict=True)
        auc = roc_auc_score(y_test, y_pred)
    
    st.markdown(f"**{model_choice} Performance**:")
    st.write(f"- **Accuracy**: {report['accuracy']:.2f}")
    st.write(f"- **AUC**: {auc:.4f}")
    st.write(f"- **Classification Report**:")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Down', 'Up'], y=['Down', 'Up'], text=cm, texttemplate="%{text}", colorscale='Blues'))
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
    st.plotly_chart(fig)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall'))
    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
    st.plotly_chart(fig)

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.markdown("Comparison of LSTM+FinBERT, Random Forest, and RNN based on accuracy, AUC, F1-score, and training time.")
    st.dataframe(comparison_df)
    
    # Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['Accuracy'], name='Accuracy'))
    fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['AUC'], name='AUC'))
    fig.add_trace(go.Bar(x=connection_df['Model'], y=comparison_df['F1-Score (Class 1)'], name='F1-Score (Class 1)'))
    fig.update_layout(title='Model Comparison', xaxis_title='Model', yaxis_title='Metric Value', barmode='group')
    st.plotly_chart(fig)
    
    st.markdown("""
    **Insights**:
    - **LSTM+FinBERT**: Likely the best performer due to its ability to model sequential numerical data and contextual news sentiment, but slowest to train.
    - **Random Forest**: Faster training, but may perform worse due to loss of temporal information from flattened features.
    - **RNN**: Simpler than LSTM, potentially lower performance due to vanishing gradients, but faster training.
    """)

elif page == "Data Visualization":
    st.header("Data Visualization")
    feature = st.selectbox("Select Feature to Plot", ['Adj Close', 'RSI', 'MACD', 'Sentiment_Positive', 'Sentiment_Negative', 'Sentiment_Neutral'])
    date_range = st.slider("Select Date Range", min_value=full_df['Date'].min(), max_value=full_df['Date'].max(), value=(full_df['Date'].min(), full_df['Date'].max()))
    
    filtered_df = full_df[(full_df['Date'] >= date_range[0]) & (full_df['Date'] <= date_range[1])]
    fig = px.line(filtered_df, x='Date', y=feature, title=f'{feature} Over Time')
    st.plotly_chart(fig)