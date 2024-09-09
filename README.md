### Developed By : Sri Varshan P
### Register No. 212222240104
### Date : 


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

### AIM:

#### To implement ARMA model in python for Russian Equipment Loss

### ALGORITHM:

1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
  data points using the ArmaProcess class. Plot the generated time series and set the title and x-
  axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
  plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
  data points using the ArmaProcess class. Plot the generated time series and set the title and x-
  axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
  plot_acf and plot_pacf.

### PROGRAM:

#### Importing necessary libraries and loading and displaying the dataset:

```py

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import sklearn.metrics
from statsmodels.tsa.arima_process import ArmaProcess

# Load the dataset
df = pd.read_csv('russia_losses_equipment.csv')

```
#### Extracting and Plotting the data the temperature data:

```py
# Extracting and cleaning the 'tank' data
X = df['tank'].replace([np.inf, -np.inf], np.nan).dropna()

# Plot the tank data
plt.figure(figsize=(10, 6))
plt.plot(X, label='Tank Data')
plt.title('Tank Data Plot')
plt.xlabel('Index')
plt.ylabel('Number of Tanks')
plt.legend()
plt.show()
```


#### Augmented Dickey-Fuller Test

```py
# Augmented Dickey-Fuller Test
dtest = adfuller(X, autolag='AIC')
print("\nAugmented Dickey-Fuller Test:")
print(f"ADF Statistic: {dtest[0]}")
print(f"p-value: {dtest[1]}")
print(f"No. of Lags Used: {dtest[2]}")
print(f"No. of Observations Used: {dtest[3]}")
```

#### Train-Test Split and Model Fitting:


```py

# Train-Test Split
train_size = len(X) - 15
X_train, X_test = X[:train_size], X[train_size:]

```

#### ARIMA Fitting

```py
# Fit ARIMA model
p, d, q = 3, 0, 2
arma_model = ARIMA(X_train, order=(p, d, q)).fit()

# Model Summary
print("\nARIMA Model Summary:")
print(arma_model.summary())
```
```py
# Fit ARMA(1,1) model to the 'tank' data to estimate parameters
arma_model = ARIMA(X, order=(1, 0, 1)).fit()
phi_estimated = arma_model.params['ar.L1']
theta_estimated = arma_model.params['ma.L1']

# Print estimated parameters
print(f"Estimated AR(1) coefficient (phi): {phi_estimated}")
print(f"Estimated MA(1) coefficient (theta): {theta_estimated}")

# Define the AR and MA parameters using the estimated values
ar_params = np.array([1, -phi_estimated])  # Note: AR terms are negated
ma_params = np.array([1, theta_estimated])

# Create an ARMA process with the estimated parameters
arma_process = ArmaProcess(ar_params, ma_params)

# Simulate the ARMA(1,1) process
n_samples = len(X)  # Number of samples similar to the original data length
np.random.seed(42)  # For reproducibility
simulated_data = arma_process.generate_sample(nsample=n_samples)

# Plotting the simulated ARMA(1,1) process
plt.figure(figsize=(10, 6))
plt.plot(simulated_data, label='Simulated ARMA(1,1) Process')
plt.title('Simulated ARMA(1,1) Process based on Tank Data')
plt.xlabel('Time')
plt.ylabel('Simulated Value')
plt.legend()
plt.show()
```
#### Autocorrelation and Partial Autocorrelation Plots and Making Predictions:

```py
# Plotting ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(X, lags=25, ax=plt.gca())
plt.title('Autocorrelation Plot')
plt.subplot(1, 2, 2)
plot_pacf(X, lags=25, ax=plt.gca())
plt.title('Partial Autocorrelation Plot')
plt.tight_layout()
plt.show()

```

#### Model Evaluation and Plotting Predictions:


```py
# Making Predictions
pred = arma_model.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, dynamic=False)

# Model Evaluation
mse = sklearn.metrics.mean_squared_error(X_test, pred)
rmse = np.sqrt(mse)
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")

# Plotting Test Data vs Predictions
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, X_test, label='Test Data')
plt.plot(X_test.index, pred, label='Predictions')
plt.title('Test Data vs Predictions')
plt.xlabel('Index')
plt.ylabel('Number of Tanks')
plt.legend()
plt.show()

```



### OUTPUT:

#### Time Series plot 

![image](https://github.com/user-attachments/assets/4d74cee3-9a24-4863-a292-5638b97e8e8c)


#### SIMULATED ARMA(1,1) PROCESS:

![image](https://github.com/user-attachments/assets/6134a34e-1812-461d-b995-a6357922b25c)


#### Autocorrelation and Partial Autocorrelation

![image](https://github.com/user-attachments/assets/b6c9cd67-cbd8-4429-938a-7e0866fb647d)



SIMULATED ARMA(2,2) PROCESS:

Partial Autocorrelation



Autocorrelation

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
