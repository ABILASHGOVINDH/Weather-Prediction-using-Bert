## Weather Report Summary & Next-Day Temperature Prediction
This project analyzes historical weather data from India (2000-2024) and builds a BERT-based model to predict the next day's maximum temperature. The notebook includes data preprocessing, feature engineering, text generation from weather features, and transformer-based regression modeling.

## 📁 Project Structure
- ├── data/                                 # Sample data (not included in repo)
- ├── notebooks/
- │   └── weather_report_summary.ipynb      # Main analysis and modeling notebook
- ├── src/                                   # Source code (if modularized)
- ├── models/                                # Saved model outputs
- ├── results/                               # Training results and metrics
- ├── logs/                                  # Training logs
- └── README.md

## 📊 Dataset
The dataset india_2000_2024_daily_weather.csv contains daily weather records for multiple Indian cities including:

- Features: city, date, temperature (max/min), apparent temperature, precipitation, rain, weather code, wind speed, wind gusts, wind direction

- Samples: 5000 randomly selected records

- Preprocessing: Normalization, encoding, and feature engineering applied

## Model Architecture
Base Model: BertForSequenceClassification from HuggingFace Transformers

Task: Regression (predict next day's max temperature)

Input: Textual summary of weather features (e.g., "city 0, month 1 weather 0, Tmin 0.399, Tmax 1.093...")

Output: Normalized next-day maximum temperature

## Required packages:

* pandas

* numpy

* matplotlib

* seaborn

* scikit-learn

* tensorflow

* torch

* transformers

* evaluate

## 🚀 Usage
Data Preparation: Place your weather data in the data/ directory

Preprocessing: Run the notebook to preprocess data and create features

Training: Execute the training cells to train the BERT model

Evaluation: Use the provided evaluation metrics to assess model performance

Prediction: Make predictions on new weather data

## 📋 Key Features
Data Normalization: StandardScaler for numerical features

Feature Engineering: Temporal features (year, month, day, day_of_year)

Text Generation: Creates textual weather summaries for BERT input

Transformer Fine-tuning: Custom BERT model for regression tasks

Evaluation Metrics: MAE, MSE, RMSE, and R² scores

## 🔮 Future Improvements
Experiment with different transformer architectures

Add more temporal features and weather lag variables

Implement hyperparameter tuning

Deploy as a web service for real-time predictions

Add more cities and weather stations to the dataset



