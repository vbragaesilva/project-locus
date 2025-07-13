import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datasets/listings_rio_1.csv')

print(df.head(10))
print(df.dtypes)
df.describe()


##############################
# Ajeitando os dados

df['price_float'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)


map_superhost = {'t': 1, 'f': 0}
df['is_superhost'] = df['host_is_superhost'].map(map_superhost)


# Mapa de calor da distribuicao do dataframe
prices_log = np.log1p(df['price_float'])  # log(1 + preço)
plt.figure(figsize=(12, 8))
sc = plt.scatter(
    df['longitude'],
    df['latitude'],
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

##############################
# FILTRANDO

df = df[df['room_type'] == 'Entire home/apt']
# df = df[df['room_type'] == 'Private room']


# df = df[df['is_superhost'] == 0]
# df = df[df['is_superhost'] == 1]

# POR BAIRRO
'''
Fiz umas regressoes sem filtrar o bairro e usar a latitute e longitude que tem no arquivo
mas o R^2 tava ficando bem ruim (nao sei direito pq), mas acho que filtrar por bairro ta ok
cada bairro tem uma precificação diferente
'''

# print(df['neighbourhood_cleansed'].value_counts())

'''
Principais bairros em quantidade
Copacabana                  6645
Barra da Tijuca             2131
Ipanema                     1770
Recreio dos Bandeirantes    1361
Jacarepaguá                 1120
'''

# Escolher o bairro
bairro = 'Copacabana'

df = df[df['neighbourhood_cleansed'] == bairro]
##############################


##############################

cols = [
    'price_float',
    'is_superhost',
    #'latitude',
    #'longitude',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'minimum_nights',
    'maximum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'availability_365'
]

# tirar os na
df_clean = df[cols].dropna()


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
'''
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
'''


df_clean = df_clean.query('minimum_nights < 20 &'
                          'price_float < 3000 &'
                          'maximum_nights > 4 &'
                          'review_scores_rating > 2 &'
                          'review_scores_accuracy > 2 &'
                          'review_scores_cleanliness > 2 &'
                          'review_scores_checkin > 2 &'
                          'review_scores_communication > 2 &'
                          'review_scores_location > 2 &'
                          'accommodates < 15 &'
                          'bedrooms < 6 &'
                          'bathrooms < 15')
df_clean.describe()


# MATRIZ DE CORRELACAO
corr = df_clean.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='RdBu', fmt='.2f', square=True, linecolor='white', annot=True)
plt.title('Matriz de Correlação')
plt.show()
