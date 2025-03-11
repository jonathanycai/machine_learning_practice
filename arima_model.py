import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# load datasets
data_train_a = pd.read_csv('data/cpu-train-a.csv', parse_dates=['datetime'])
data_test_a = pd.read_csv('data/cpu-test-a.csv', parse_dates=['datetime'])

# visualize training data
plt.figure(figsize=(16,8))
plt.plot(data_train_a['datetime'], data_train_a['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization - Train A')
plt.show()

# fit ARIMA model to dataset A (AR = 11, MA = 11, I = 0)
model_a = ARIMA(data_train_a['cpu'], order=(11,0,11))
model_a_fit = model_a.fit(method_kwargs={"maxiter": 200})
print(model_a_fit.summary())

# plot fitted values vs original training data
plt.figure(figsize=(20, 8))
plt.plot(data_train_a['datetime'], data_train_a['cpu'], color='black', label='Original')
plt.plot(data_train_a['datetime'], model_a_fit.fittedvalues, color='blue', alpha=0.7, label='Fitted')
plt.ylabel('CPU %')
plt.title('ARIMA Model Fit - Dataset A')
plt.legend()
plt.show()

# forecast future values
forecast_steps = 60
forecast_a = model_a_fit.get_forecast(steps=forecast_steps)
forecast_index_a = pd.date_range(start=data_train_a['datetime'].iloc[-1], periods=forecast_steps+1, freq='H')[1:]