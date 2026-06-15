import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sl = pd.read_csv("Walmart_Sales.csv")

print(sl.head())
print("\nDataset Information:")
print(sl.info())
print("\nNull Values:")
print(sl.isnull().sum())
print("\nDuplicates:", sl.duplicated().sum())

sl['Date'] = pd.to_datetime(sl['Date'], format='%d-%m-%Y')

print("\nDate Data Type:", sl['Date'].dtype)
print("Dataset Start Date:", sl['Date'].min())
print("Dataset End Date:", sl['Date'].max())
print("Number of Stores:", sl['Store'].nunique())


sales_trend = sl.groupby('Date')['Weekly_Sales'].sum().reset_index()

print("\nSales Trend Dataset:")
print(sales_trend.head())

plt.figure(figsize=(12,6))
sns.lineplot(data=sales_trend,x='Date',y='Weekly_Sales')
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.show()

holiday_sales = sl.groupby('Holiday_Flag')['Weekly_Sales'].mean()

plt.figure(figsize=(8,5))
plt.bar(holiday_sales.index, holiday_sales.values)
plt.title("Average Sales: Holiday vs Non-Holiday")
plt.xlabel("Holiday Flag")
plt.ylabel("Average Weekly Sales")
plt.show()

print("\nHoliday Impact:")
print(holiday_sales)


corr = sl[['Weekly_Sales','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment']].corr()

plt.figure(figsize=(8,6))

sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

print("\nCorrelation Matrix:")
print(corr)

ndata = sl.groupby('Date')['Weekly_Sales'].sum().reset_index()

print("\nForecasting Dataset:")
print(ndata.head())
print("Shape:", ndata.shape)

ndata["Moving_Average"] = (ndata["Weekly_Sales"].rolling(window=4).mean())

plt.figure(figsize=(12,6))

plt.plot(ndata["Date"],ndata["Weekly_Sales"],label="Actual Sales")

plt.plot(ndata["Date"],ndata["Moving_Average"],label="4-Week Moving Average")

plt.title("Weekly Sales vs Moving Average")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.show()

# Naive Forecast

ndata["Naive_Forecast"] = ndata["Weekly_Sales"].shift(1)

evaluation_data = ndata.dropna()

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

mae = mean_absolute_error(evaluation_data['Weekly_Sales'],evaluation_data['Naive_Forecast'])

rmse = np.sqrt(mean_squared_error(evaluation_data['Weekly_Sales'],evaluation_data['Naive_Forecast']))

print("\nNaive Forecast Performance")
print("MAE:", mae)
print("RMSE:", rmse)

# Train & Test split

split_index = int(len(ndata) * 0.8)

train_data = ndata[:split_index]
test_data = ndata[split_index:]


# Model initialization

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(
    train_data['Weekly_Sales'],
    order=(1,1,1)
)

model_fit = model.fit()

print("\nARIMA Model Summary")
print(model_fit.summary())

# Forecast

forecast = model_fit.forecast(
    steps=len(test_data)
)

print("\nForecast Sample:")
print(forecast.head())

# ARIMA

mae_arima = mean_absolute_error(
    test_data['Weekly_Sales'],
    forecast
)

rmse_arima = np.sqrt(
    mean_squared_error(
        test_data['Weekly_Sales'],
        forecast
    )
)

print("\nARIMA Performance")
print("MAE:", mae_arima)
print("RMSE:", rmse_arima)

# Actual Vs Forecast

plt.figure(figsize=(12,6))

plt.plot(
    test_data['Date'],
    test_data['Weekly_Sales'],
    label='Actual Sales'
)

plt.plot(
    test_data['Date'],
    forecast,
    label='ARIMA Forecast'
)

plt.title('Actual Sales vs ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')

plt.legend()

plt.show()

# Model Comparison

comparison = pd.DataFrame({
    'Model': ['Naive Forecast', 'ARIMA'],
    'MAE': [mae, mae_arima],
    'RMSE': [rmse, rmse_arima]
})

print("\nModel Performance Comparison")
print(comparison)