# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:48:57 2020

@author: vijayk1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('C:/Users/itadmin/Desktop/Live_Proj/Google_Stock_Price_Prediction/GOOGL.csv',parse_dates=True)
df_train = df[df['Date'] < '2019-01-01' ].copy()
df_test = df[df['Date'] > '2019-01-01'].copy()
df_tr = df_train.drop(['Date','Adj Close'],axis=1)
scaler = MinMaxScaler()
df_tr_std = scaler.fit_transform(df_tr)

# X_train and y_Train
# Here we are taking bunch of 60 days and predicting the 61th day, then we take 
# from 1 to 61 days as training and 62nd day as prediction and so on

X_train = []
y_train = []
for i in range(60,df_tr_std.shape[0]):
    X_train.append(df_tr_std[i-60:i])
    y_train.append(df_tr_std[i,0])

X_train, y_train = np.array(X_train),np.array(y_train)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
regressor = Sequential()
# Ist LSTM Cell with 50 units and 0.2 dropout
regressor.add(LSTM(units = 50,activation='relu',return_sequences=True,input_shape=(X_train.shape[1],5)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60,activation='relu',return_sequences=True)
# regressor.add(Dropout(0.3))
# regressor.add(LSTM(units = 80,activation='relu',return_sequences=True)
# regressor.add(Dropout(0.4))
# regressor.add(LSTM(units = 120,activation='relu')
# regressor.add(Dropout(0.5))
regressor.add(Dense(units=1))
regressor.summary()
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(X_train,y_train,epochs=10,batch_size=32)

# Lets check the prediction
last_60_tr = df_train.tail(60)
df_test = last_60_tr.append(df_test)

df_test = df_test.drop(['Date','Adj Close'],axis=1)
scaler = MinMaxScaler()
df_test_std = scaler.fit_transform(df_test)
X_test = []
y_test = []
for i in range(60,df_test_std.shape[0]):
    X_test.append(df_test_std[i-60:i])
    y_test.append(df_test_std[i,0])

X_test, y_test = np.array(X_test),np.array(y_test)

y_pred = regressor.predict(X_test)
scale = 1/1.66469671e-03
y_pred = y_pred*scale
y_test = y_test*scale

# Visualisation
plt.figure(figsize=(14,5))
plt.plot(y_test,color='red',label='Google_Actual_Stock_Price')
plt.plot(y_pred[:,0,:],color='blue',label='Google_Predicted_Stock_Price')
plt.legend()
plt.show()

