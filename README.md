# 🌟 Pharma Sales Forecasting App

This project predicts **daily**, **weekly**, and **monthly** sales for pharmaceutical drugs using machine learning.  
The dataset spans **2014 to 2021**, containing hourly consumption values of multiple drug categories.  
The final solution uses **XGBoost** for prediction and is deployed using **Streamlit**.

The application helps manufacturing companies predict drug demand, plan inventory, and optimize production.

## 🚀 Features

### 🔹 1. Predict sales for any drug  
Supports these drug categories:
- M01AB  
- M01AE  
- N02BA  
- N02BE  
- N05B  
- N05C  
- R03  
- R06  

### 🔹 2. Monthly Forecast  
Predicts **total expected sales** for a selected month and year.

### 🔹 3. Weekly Forecast (Fixed 4-Week Segments)
- Week 1 → Days 1–7  
- Week 2 → Days 8–14  
- Week 3 → Days 15–21  
- Week 4 → Days 22–end  

### 🔹 4. Daily Forecast  
Detailed day-by-day prediction line chart.

### 🔹 5. Clean Streamlit UI  
Interactive dropdowns, charts, and summaries.

---

## 🧹 Data Preprocessing

Several preprocessing steps were applied:

### ✔ Handling Missing Values  
Dataset verified for missing dates and values. Minor inconsistencies corrected.

### ✔ Feature Engineering  
From the `datum` timestamp, new features were created:
- `Year`
- `Month`
- `Hour`
- `Weekday Name`
- `Drug` (categorical label)

### ✔ Label Encoding  
Converted categorical columns to numeric form:
- `Weekday Name`
- `Drug`

### ✔ Outlier Detection  
Using:
- Z-Score  
- Boxplots  
- Domain knowledge  

Extreme spikes removed to improve model stability.

### ✔ Train-Test Split  
Data split chronologically to avoid leakage and maintain time-series correctness.

---

## 📊 Exploratory Data Analysis (EDA)

Comprehensive EDA was performed, including:

### ✔ Time Series Analysis  
- Daily, weekly, and monthly plots  
- Seasonality detection  
- Hour-wise trends  

### ✔ ACF & PACF  
Used to understand autocorrelation and lag dependencies.

### ✔ Distribution Analysis  
- Log transformation for right-skewed drug consumption  
- Histograms & KDE plots  
- Boxplots for variability and outlier detection

### ✔ Correlation Study  
Correlation heatmaps used to uncover relationships between drugs.

EDA insights helped engineer useful features and select appropriate models.

---

## 🤖 Models Used & Compared

Multiple ML models were trained and evaluated:

### ✔ 1. Random Forest Regressor  
- Good baseline  
- Handles non-linearity  
- Medium accuracy  

### ✔ 2. XGBoost Regressor (Best Model)
- Best MAE & RMSE  
- Excellent for non-linear patterns  
- Supports early stopping  
- Final chosen model for deployment  

### ✔ 3. LSTM Neural Network  
- Captures sequence patterns  
- Requires more training  
- Did not outperform XGBoost  

### ✔ 4. ARIMA / SARIMA  
- Suitable only for pure time-series  
- High error on multi-variable dataset  
- Not chosen  

---

## 🏆 Final Model Selection: XGBoost

### 📌 Why XGBoost?
- Lowest prediction error  
- Handles categorical + numerical + time features  
- Prevents overfitting with early stopping  
- Very fast for inference  
- Overall strongest performer

---

## 📈 Model Evaluation Summary

| Model          | MAE     | RMSE    | Notes                       |
|----------------|---------|---------|------------------------------|
| Random Forest  | Medium  | Medium  | Good baseline                |
| **XGBoost**    | **Low** | **Low** | **Selected model**           |
| LSTM           | Medium  | High    | Needs tuning                 |
| ARIMA          | High    | High    | Not suitable                 |

---

## 🌐 Streamlit App

The Streamlit app allows the user to:

### ✔ Select drug  
### ✔ Select year & month  
### ✔ Get monthly total prediction  
### ✔ View weekly breakdown  
### ✔ View daily forecast line chart  

Run the app using:

```bash
streamlit run app.py

