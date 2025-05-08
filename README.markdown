# DLP_Project: Stock Price Prediction Using Hybrid Machine Learning Models

## Project Overview
This project focuses on predicting the directional movement (up or down) of the Dow Jones Industrial Average (DJIA) stock index using a hybrid approach that combines numerical time-series data and textual news sentiment. Three models are implemented and compared: LSTM with FinBERT, Random Forest, and a Simple RNN. The project includes data preprocessing, model training, evaluation, and a detailed report.

### Objectives
- Integrate numerical stock data with news sentiment for enhanced prediction accuracy.
- Compare hybrid deep learning models (LSTM+FinBERT, RNN) with a traditional machine learning model (Random Forest).
- Provide detailed evaluation metrics for model performance.

## Dataset
The project uses the following datasets:
- **Combined_News_DJIA.csv**: Daily news headlines aligned with DJIA dates.
- **upload_DJIA_table.csv**: Historical DJIA stock prices (2008–2016).
- **RedditNews.csv**: Additional news data (not used in the final merge).

## Files in the Repository
- **model_train.py**: Main script for data preprocessing, model training, and evaluation. Outputs detailed metrics (classification report, AUC, confusion matrix, precision-recall curve) in Colab.
- **Stock_Price_Prediction_Project_Report.tex**: LaTeX report documenting the project methodology, results, and analysis.
- **StockPrediction_Model/**: Directory containing saved models and data:
  - `best_model.keras`, `final_model.keras`: LSTM+FinBERT models.
  - `rnn_model.keras`, `rnn_final_model.keras`: RNN models.
  - `rf_model.pkl`: Random Forest model.
  - `model_comparison.csv`: Model performance comparison.
  - `preprocessed_data.csv`, `.npy` files: Preprocessed data and test encodings.
- **datasets/**: Directory containing the raw datasets (`Combined_News_DJIA.csv`, `upload_DJIA_table.csv`, `RedditNews.csv`).

## Model Performance
The models were evaluated on a test set (2015-01-01 onward) with the following results:

| Model         | Accuracy | AUC    | F1-Score (Class 1) | Training Time (s) |
|---------------|----------|--------|--------------------|-------------------|
| LSTM+FinBERT  | 0.486    | 0.500  | 0.465              | 763.54            |
| Random Forest | 0.472    | 0.490  | 0.499              | 6.78              |
| RNN           | 0.497    | 0.482  | 0.627              | 1124.41           |

- **RNN** achieved the highest F1-score for the positive class (0.627), indicating better sensitivity to "up" movements.
- All models have AUC values near 0.5, suggesting limited predictive power with the current dataset and features.

## Prerequisites
To run this project, ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.12.0
- Transformers (Hugging Face)
- Scikit-learn
- Pandas, NumPy
- Joblib
- Google Colab (for GPU support)

Install dependencies using:
```bash
pip install tensorflow==2.12.0 transformers scikit-learn pandas numpy joblib
```

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/DLP_Project.git
   cd DLP_Project
   ```

2. **Set Up Google Drive**:
   - The script uses Google Drive to save models. Mount your Google Drive in Colab or adjust the `drive_model_path` to a local directory.

3. **Upload Datasets**:
   - Place `Combined_News_DJIA.csv`, `upload_DJIA_table.csv`, and `RedditNews.csv` in the `datasets/` directory or upload them to your Colab environment.

4. **Run the Script in Colab**:
   - Open `model_train.py` in Google Colab.
   - Execute the script to preprocess data, train models, and view evaluation metrics in the output.
   - The script will save models and data to `StockPrediction_Model/` on your Google Drive.

5. **View Results**:
   - Detailed metrics (classification report, AUC, confusion matrix, precision-recall curve) for each model are printed in the Colab output.
   - Check `StockPrediction_Model/model_comparison.csv` for a summary of model performance.

## Project Report
A detailed report is provided in `Stock_Price_Prediction_Project_Report.tex`. Compile it using a LaTeX editor (e.g., Overleaf) to generate a PDF with the methodology, results, and analysis.

## Future Improvements
- Enhance feature engineering by adding more technical indicators or sentiment features.
- Experiment with a larger dataset or different time windows.
- Tune hyperparameters (e.g., learning rate, batch size) for better performance.

## Acknowledgments
- The FinBERT model is sourced from `yiyanghkust/finbert-tone` (Huang & Yang, 2020).
- This project was developed as part of a deep learning course evaluation.

## License
This project is licensed under the MIT License.