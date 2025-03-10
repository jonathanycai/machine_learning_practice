import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import warnings
from prophet import Prophet

data = pd.read_csv('data/GOOG.csv')
print(data.head())

# plot data
plt.style.use("fivethirtyeight")
plt.figure(figsize=(16,8))
plt.title("Google Closing Stock Price")
plt.plot(data["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()

data = data[["Date", "Close"]]
data = data.rename(columns={"Date": "ds", "Close": "y"})
print(data.head())

m = Prophet(daily_seasonality=True)
m.fit(data)

future = m.make_future_dataframe(periods=365)
predictions = m.predict(future)

fig1 = m.plot(predictions)
plt.title("Prediction of GOOGLE Stock Price")
plt.xlabel("Date")
plt.ylabel("Closing Stock Price")
plt.show()

fig2 = m.plot_components(predictions)
plt.show()