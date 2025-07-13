import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from exame_ler import df_clean


# Separando
X = df_clean.drop('price_float', axis=1)
Y = df_clean['price_float']

# NORMALIZACAO
min_max_scaler = StandardScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=26)


#####################
# 1) KNN REGRESSOR
#####################
from sklearn.neighbors import KNeighborsRegressor
# KNN
# Ajuste (fit) do modelo aos dados de treino
model1 = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
model1.fit(X_train, Y_train)
# Predição nos dados de teste
Y_pred = model1.predict(X_test)
# Cálculo das métricas de erro:
# r2: coeficiente de determinação (quanto mais próximo de 1, melhor)
# mae: erro absoluto médio
# mse: erro quadrático médio
# rmse: raiz do erro quadrático médio
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('KNN Regressor')
print(r2)
print(mae, mse, rmse)
print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {np.mean(Y_pred-Y_test)}')


#####################
# 3) Support Vector Machines
#####################
from sklearn.svm import SVR
# SVM
# Ajuste (fit) do modelo aos dados de treino
model3 = SVR()
model3.fit(X_train, Y_train)
# Predição nos dados de teste
Y_pred = model3.predict(X_test)
# Cálculo das métricas de erro
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('Support Vector Machines')
print(r2)
print(mae, mse, rmse)


#####################
# 4) Random Forest
#####################
from sklearn.ensemble import RandomForestRegressor
# Random Forest
# Ajuste (fit) do modelo aos dados de treino
model5 = RandomForestRegressor(n_estimators=100, random_state=26)
model5.fit(X_train, Y_train)
# Predição nos dados de teste
Y_pred = model5.predict(X_test)
# Cálculo das métricas de erro
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('random forest')
print(r2)
print(mae, mse, rmse)

