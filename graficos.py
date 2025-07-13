import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from exame_ler import df_clean, df0


# Mapa de calor da distribuicao do dataframe
prices_log = np.log1p(df0['price_float'])  # log(1 + preço)
plt.figure(figsize=(12, 8))
sc = plt.scatter(
    df0['longitude'],
    df0['latitude'],
    c=prices_log,
    cmap='jet',
    s=8,
    alpha=0.4
)
plt.colorbar(sc, label='Preço')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Preço por Localização')
plt.show()

##############################

# Hist dos dados
df_clean.hist(bins=15, figsize=(12, 8))
plt.show()
print(df_clean.describe())

# Tirando os outliers
plt.figure(figsize=(10, 3))
sns.boxplot(data=df_clean, x='minimum_nights')
plt.title('Boxplot minimum_nights')
# plt.show()
print(f'[minimun_nights]\nValores acima de 30: {len(df_clean[df_clean.minimum_nights > 20])} entradas')
print('Porcentagem: {:.4f}%'.format(len(df_clean[df_clean.minimum_nights > 20])/len(df_clean.minimum_nights)* 100))

plt.figure(figsize=(10, 3))
sns.boxplot(data=df_clean, x='price_float')
plt.title('Boxplot price_float')
# plt.show()
print(f'[price]\nValores acima de 3000: {len(df_clean[df_clean.price_float > 3000])} entradas')
print('Porcentagem: {:.4f}%'.format(len(df_clean[df_clean.price_float > 3000])/len(df_clean.price_float) * 100))

plt.figure(figsize=(10, 3))
sns.boxplot(data=df_clean, x='maximum_nights')
plt.title('Boxplot maximum_nights')
# plt.show()
print(f'[maximum_nights]\nValores abaixo de 3: {len(df_clean[df_clean.maximum_nights < 5])} entradas')
print('Porcentagem: {:.4f}%'.format(len(df_clean[df_clean.maximum_nights < 5])/len(df_clean.maximum_nights)* 100))

# review
plt.figure(figsize=(10, 3))
#sns.boxplot(data=df_clean, x='review_scores_rating')
plt.title('Boxplot review_scores_rating')
#plt.show()

print(f'[review_scores_rating]\nValores abaixo de 2: {len(df_clean[df_clean.review_scores_rating <= 2])} entradas')
print('{:.4f}%'.format(len(df_clean[df_clean.review_scores_rating <= 2])/len(df_clean.review_scores_rating)* 100))

plt.figure(figsize=(10, 3))
#sns.boxplot(data=df_clean, x='accommodates')
plt.title('Boxplot accommodates')
#plt.show()
print('{:.4f}%'.format(len(df_clean[df_clean.accommodates > 14])/len(df_clean.accommodates)* 100)) # ~1.1%

plt.figure(figsize=(10, 3))
#sns.boxplot(data=df_clean, x='bedrooms')
plt.title('Boxplot bedrooms')
#plt.show()
print('{:.4f}%'.format(len(df_clean[df_clean.bedrooms > 5])/len(df_clean.bedrooms)* 100)) # ~0.8%

plt.figure(figsize=(10, 3))
#sns.boxplot(data=df_clean, x='bathrooms')
plt.title('Boxplot bathrooms')
#plt.show()
print('{:.4f}%'.format(len(df_clean[df_clean.bathrooms > 6])/len(df_clean.bathrooms)* 100)) # ~1.2%

plt.figure(figsize=(10, 3))
sns.boxplot(data=df_clean, x='beds')
plt.title('Boxplot beds')
plt.show()
print('{:.4f}%'.format(len(df_clean[df_clean.beds > 15])/len(df_clean.beds)* 100)) # 0.45%


# MATRIZ DE CORRELACAO
corr = df_clean.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='RdBu', fmt='.2f', square=True, linecolor='white', annot=True)
plt.title('Matriz de Correlação')
plt.show()
