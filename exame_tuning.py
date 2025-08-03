import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from exame_ler import final_df


# Separando
X = final_df.drop('price_float', axis=1)
Y = final_df['price_float']

# NORMALIZACAO
min_max_scaler = StandardScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=28)


#####################
# 1) KNN REGRESSOR
#####################


# Otimizando KNN com validação cruzada e redução de dimensionalidade (PCA)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

# Redução de dimensionalidade com PCA:
# O PCA reduz o número de variáveis do problema, mantendo 95% da variância dos dados.
# Isso acelera o cálculo das distâncias no KNN e pode melhorar a generalização.
pca = PCA(n_components=0.95, random_state=26)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Busca dos melhores hiperparâmetros do KNN:
# Utiliza GridSearchCV, que testa exaustivamente todas as combinações possíveis dos parâmetros fornecidos.
# Testamos diferentes valores de k (n_neighbors), tipos de peso e métricas de distância.
param_grid_knn = {
    'n_neighbors': list(range(2, 21)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(
    KNeighborsRegressor(),
    param_grid=param_grid_knn,
    scoring='neg_mean_squared_error', # métrica de avaliação
    cv=5, # 5-fold cross-validation
    n_jobs=-1 # usa todos os núcleos disponíveis
)
grid_knn.fit(X_train_pca, Y_train)
Y_pred = grid_knn.predict(X_test_pca)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('KNN Regressor (GridSearchCV + PCA)')
print('Best params:', grid_knn.best_params_)
print(r2)
print(mae, mse, rmse)



# SVR com RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

# Busca dos melhores hiperparâmetros do SVR:
# Utiliza RandomizedSearchCV, que sorteia combinações aleatórias dos parâmetros dentro dos intervalos fornecidos.
# Isso é mais eficiente que o GridSearchCV quando há muitos parâmetros ou valores possíveis.
param_dist_svr = {
    'C': np.logspace(-2, 2, 10),
    'gamma': np.logspace(-3, 1, 10),
    'epsilon': np.linspace(0.01, 1, 10),
    'kernel': ['rbf']
}
random_search_svr = RandomizedSearchCV(
    SVR(),
    param_distributions=param_dist_svr,
    n_iter=20, # número de combinações testadas
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=28,
    n_jobs=-1
)
random_search_svr.fit(X_train, Y_train)
Y_pred = random_search_svr.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('Support Vector Machines (RandomizedSearchCV)')
print('Best params:', random_search_svr.best_params_)
print(r2)
print(mae, mse, rmse)


# Random Forest com RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
# Busca dos melhores hiperparâmetros do Random Forest:
# Também utiliza RandomizedSearchCV, testando combinações aleatórias dos parâmetros.
# Isso permite explorar um espaço maior de possibilidades sem testar todas as combinações.
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
random_search_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=26),
    param_distributions=param_dist_rf,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=28,
    n_jobs=-1
)
random_search_rf.fit(X_train, Y_train)
Y_pred = random_search_rf.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('Random Forest (RandomizedSearchCV)')
print('Best params:', random_search_rf.best_params_)
print(r2)
print(mae, mse, rmse)


