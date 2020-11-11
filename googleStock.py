import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

data = pd.read_csv("GOOG.csv") 
print(data.head(5))
print(data.describe())

data = data[["Date","Close"]] 
data = data.rename(columns = {"Date":"ds","Close":"y"}) 
print(data.head(5))

m = Prophet(daily_seasonality = True) 
m.fit(data) 

future = m.make_future_dataframe(periods=365) 
prediction = m.predict(future)
m.plot(prediction)
plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()

m.plot_components(prediction)
plt.show()