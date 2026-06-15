 # Walmart Sales Forecasting (TSA)

 ## Intern ID: CITS3946

A Time Series Forecasting project using the Walmart Sales dataset downloaded from kaggle. This project focuses on data cleaning, EDA, trend analysis, forecasting, and model evaluation using Python.

---

## Project Overview

The objective of this project is to analyze historical Walmart weekly sales data and build a forecasting model capable of predicting future sales trends.

The project includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Sales Trend Analysis
- Feature Impact Analysis
- Correlation Analysis
- Moving Average Analysis
- ARIMA Time Series Forecasting
- Forecast Evaluation using MAE and RMSE

---

## Dataset Information

The dataset contains weekly sales records from multiple Walmart stores along with economic and environmental factors such as:

- Store Number
- Date
- Weekly Sales
- Holiday Flag
- Temperature
- Fuel Price
- Consumer Price Index (CPI)
- Unemployment Rate

Link: https://www.kaggle.com/datasets/mikhail1681/walmart-sales

### Dataset Summary

- Total Records: 6,435
- Stores: 45
- Time Period: February 2010 – October 2012
- Missing Values: None
- Duplicate Records: None

---

## Technologies Used

###Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-Learn
- Statsmodels

---

## Exploratory Data Analysis

### Sales Trend Analysis

A time-series visualization was created to analyze weekly sales performance across every store.

### Holiday Impact Analysis

Average weekly sales during holiday and non-holiday periods were compared to identify the influence of holidays on revenue generation.

### Correlation Analysis

A correlation heatmap was generated to study relationships between:

- Weekly Sales
- Holiday Flag
- Temperature
- Fuel Price
- CPI
- Unemployment

### Key Findings

- Significant sales spikes were observed during end of year periods.
- Holiday weeks made higher average sales than non-holiday weeks.
- Economic indicators showed weak correlations with weekly sales.
- Unemployment exhibited the strongest negative relationship with sales among the analyzed variables.

---

## Forecasting Workflow

### 1. Data Aggregation

Weekly sales from all stores were aggregated by date to create a forecasting dataset.

### 2. Moving Average

A 4-week moving average was applied to smooth short-term fluctuations and identify long-term trends.

### 3. Naive Forecast

A baseline forecasting model was created using the previous week's sales as the prediction for the current week.

### 4. ARIMA Model

An ARIMA(1,1,1) model was trained on historical sales data to forecast future weekly sales.

---

## Model Evaluation

### Naive Forecast Performance

| Metric | Value |
|----------|----------|
| MAE | 3,178,912 |
| RMSE | 6,305,365 |

### ARIMA(1,1,1) Performance

| Metric | Value |
|----------|----------|
| MAE | 1,826,656 |
| RMSE | 2,331,799 |

---

## Performance Comparison

| Model | MAE | RMSE |
|----------|----------|----------|
| Naive Forecast | 3,178,912 | 6,305,365 |
| ARIMA(1,1,1) | 1,826,656 | 2,331,799 |

The ARIMA model significantly outperformed the baseline forecasting approach, reducing both forecasting error metrics and providing more reliable future sales estimates.

---

## Visualizations (Viz)

### Included Visualizations

- Total Weekly Sales Over Time
- Holiday vs Non-Holiday Sales Analysis
- Correlation Heatmap
- Weekly Sales vs Moving Average
- Actual Sales vs ARIMA Forecast

---

## Repository Structure

```text
Walmart_Sales_Forecasting/
│
└── Dataset [Kaggle]/
    ├── Walmart_Sales.csv
├── sales_fc.py
├── README.md
├── LICENSE
│
└── Viz/
    ├── final_arima_plot.png
    ├── heat_map.png
    ├── holiday_flag.png
    ├── total_sales.png
    └── weekly_avg_vs_4_week.png
```

---

## Conclusion

This project demonstrates a complete time series forecasting workflow, beginning with data exploration and ending with predictive modeling. The ARIMA(1,1,1) model successfully captured underlying sales patterns and substantially improved forecasting accuracy compared to the naive baseline model.

The analysis highlights the impact of seasonal sales behavior and demonstrates how time series forecasting techniques can support data-driven business decision-making.

---
