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

#### Importing Necessary Libraries and Loading the Dataset

```py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('russia_losses_equipment.csv')

```

####  Extracting and Plotting the 'Tank' Data

```py
# Extracting and cleaning the 'tank' data
X = df['tank'].replace([np.inf, -np.inf], np.nan).dropna()

# Plot ACF and PACF of the original data before any ARMA modeling
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(X, lags=25, ax=plt.gca())
plt.title('Original Tank Data ACF')
plt.subplot(1, 2, 2)
plot_pacf(X, lags=25, ax=plt.gca())
plt.title('Original Tank Data PACF')
plt.tight_layout()
plt.show()

```
#### Train-Test Split and ARMA Model Fitting

```py
# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]

# Fit ARMA(1,1) model to the 'tank' data to estimate parameters
arma11_model = ARIMA(X_train, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) process
ar_params_11 = np.array([1, -phi1_arma11])  # AR(1) parameter
ma_params_11 = np.array([1, theta1_arma11])  # MA(1) parameter
arma11_process = ArmaProcess(ar_params_11, ma_params_11)

# Set random seed and simulate ARMA(1,1)
np.random.seed(42)  # For reproducibility
simulated_arma11 = arma11_process.generate_sample(nsample=len(X))

```

#### Plotting Simulated ARMA(1,1) Data

```py
# Plot ACF, PACF, and Simulated ARMA(1,1)
plt.figure(figsize=(18, 6))

# Simulated ARMA(1,1) ACF and PACF
plt.subplot(1, 3, 1)
plot_acf(simulated_arma11, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(1,1) ACF')
plt.subplot(1, 3, 2)
plot_pacf(simulated_arma11, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(1,1) PACF')

# Simulated ARMA(1,1) plot
plt.subplot(1, 3, 3)
plt.plot(simulated_arma11, label='Simulated ARMA(1,1)', color='orange')
plt.title('Simulated ARMA(1,1) Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


```

#### ARMA(2,2) Model Fitting and Simulation

```py

# Fit ARMA(2,2) model to the 'tank' data to estimate parameters
arma22_model = ARIMA(X_train, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) process
ar_params_22 = np.array([1, -phi1_arma22, -phi2_arma22])  # AR(2) parameters
ma_params_22 = np.array([1, theta1_arma22, theta2_arma22])  # MA(2) parameters
arma22_process = ArmaProcess(ar_params_22, ma_params_22)

# Set random seed and simulate ARMA(2,2)
np.random.seed(42)  # For reproducibility
simulated_arma22 = arma22_process.generate_sample(nsample=len(X))


```

#### Plotting Simulated ARMA(2,2) Data

```py
# Plot ACF, PACF, and Simulated ARMA(2,2)
plt.figure(figsize=(18, 6))

# Simulated ARMA(2,2) ACF and PACF
plt.subplot(1, 3, 1)
plot_acf(simulated_arma22, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(2,2) ACF')
plt.subplot(1, 3, 2)
plot_pacf(simulated_arma22, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(2,2) PACF')

# Simulated ARMA(2,2) plot
plt.subplot(1, 3, 3)
plt.plot(simulated_arma22, label='Simulated ARMA(2,2)', color='green')
plt.title('Simulated ARMA(2,2) Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

```


### OUTPUT:

#### Autocorrelation and Partial Autocorrelation

![image](https://github.com/user-attachments/assets/64bda9f9-446b-4512-8cc9-b4e166bf01bb)



#### SIMULATED ARMA(1,1) PROCESS: Autocorrelation and Partial Autocorrelation


![image](https://github.com/user-attachments/assets/8752ceb6-e67d-4664-9eb4-e7f79cdbc232)




SIMULATED ARMA(2,2) PROCESS: Autocorrelation and Partial Autocorrelation


![image](https://github.com/user-attachments/assets/8384178a-6933-4560-88ab-0572f1cca42c)



### RESULT:

#### Thus, a python program is created to for ARMA Model successfully.
