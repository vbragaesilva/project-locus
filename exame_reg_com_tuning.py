import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from exame_ler import df_clean


X = df_clean.drop('price_float', axis=1)
Y = df_clean['price_float']

# Normalizando os dados
min_max_scaler = StandardScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=26)


''' 
KNN REGRESSOR 
'''


from sklearn.neighbors import KNeighborsRegressor
import time

# KNN
# Definição dos hiperparâmetros do modelo KNN:
# n_neighbors: número de vizinhos considerados
# metric: métrica de distância (aqui, euclidiana)
# weights: ponderação dos vizinhos (aqui, distância inversa)
model1 = KNeighborsRegressor(
    n_neighbors=20,
    metric='euclidean',
    weights='distance'
)

# Cálculo do tempo de treinamento (fit):
start_train = time.time()
model1.fit(X_train, Y_train)
end_train = time.time()
train_time = end_train - start_train

# Cálculo do tempo de inferência (predição):
start_pred = time.time()
Y_pred = model1.predict(X_test)
end_pred = time.time()
pred_time = end_pred - start_pred

# Cálculo das métricas de erro:
# r2: coeficiente de determinação
# mae: erro absoluto médio
# mse: erro quadrático médio
# rmse: raiz do erro quadrático médio
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('KNN Regressor')
print(f"Tempo de treino: {train_time:.4f}s | Tempo de inferência: {pred_time:.4f}s")
print(r2)
print(mae, mse, rmse)
print("-"*50)


'''
Support Vector Machines
'''
from sklearn.svm import SVR
# SVM
# Definição dos hiperparâmetros do SVR:
# kernel: tipo de função de kernel (rbf = radial basis function)
# C: parâmetro de regularização
# gamma: parâmetro do kernel
# epsilon: margem de tolerância para erro
model3 = SVR(kernel='rbf', C=100.0, gamma=0.0027825594022071257, epsilon=0.23)

# Cálculo do tempo de treinamento (fit):
start_train = time.time()
model3.fit(X_train, Y_train)
end_train = time.time()
train_time = end_train - start_train

# Cálculo do tempo de inferência (predição):
start_pred = time.time()
Y_pred = model3.predict(X_test)
end_pred = time.time()
pred_time = end_pred - start_pred

# Cálculo das métricas de erro:
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('Support Vector Machines')
print(f"Tempo de treino: {train_time:.4f}s | Tempo de inferência: {pred_time:.4f}s")
print(r2)
print(mae, mse, rmse)
print("-"*50)


'''
Random Forest
'''
from sklearn.ensemble import RandomForestRegressor
# Random Forest
# Definição dos hiperparâmetros do Random Forest:
# n_estimators: número de árvores na floresta
# min_samples_split: número mínimo de amostras para dividir um nó
# min_samples_leaf: número mínimo de amostras em uma folha
# max_features: número de features consideradas em cada split
# max_depth: profundidade máxima da árvore
model5 = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='log2',
    max_depth=30,
    random_state=28
)

# Cálculo do tempo de treinamento (fit):
start_train = time.time()
model5.fit(X_train, Y_train)
end_train = time.time()
train_time = end_train - start_train

# Cálculo do tempo de inferência (predição):
start_pred = time.time()
Y_pred = model5.predict(X_test)
end_pred = time.time()
pred_time = end_pred - start_pred

# Cálculo das métricas de erro:
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('random forest')
print(f"Tempo de treino: {train_time:.4f}s | Tempo de inferência: {pred_time:.4f}s")
print(r2)
print(mae, mse, rmse)
print("-"*50)

