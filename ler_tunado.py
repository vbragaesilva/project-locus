import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from typing import List

def load_dataframe(path:str):
    return pd.read_csv(path)

def format_dataframe(df: pd.DataFrame):
    # formata os dados
    # remover simbolos
    df['price_float'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    # substituir t/f por 1/0
    map_superhost = {'t': 1, 'f': 0}
    df['is_superhost'] = df['host_is_superhost'].map(map_superhost)
    return df

def clean_dataframe(df: pd.DataFrame):
    # remove as linhas NA e seleciona as colunas consideradas relevantes
    # Há espaço para considerações: 
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
    clean_df = df[cols].dropna()

    return clean_df

def filter_by_room_type(df: pd.DataFrame, room_type:str):
    # nosso interesse eh room_type='Entire home/apt'
    return df[df['room_type'] == room_type]

def filter_by_neighborhood(df: pd.DataFrame, bairro:str):
    return df[df['neighbourhood_cleansed'] == bairro]

def mapa_de_calor(df: pd.DataFrame):
    # Mapa de calor da distribuicao dos precos do dataframe
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

def histograma(df: pd.DataFrame):
    # plota o histograma, o dataframe já precisa estar limpo
    df.hist(bins=15, figsize=(12, 8))
    plt.show()



#######################
# Esse aqui está um pouco arbitrário
# Ainda nao sei se vamos manter o removedor de outliers, talvez um menos "agressivo" que so remova os muito zoados
# Acho também que é uma boa padronizar remover somente os 1-2% mais "foras"
def remove_outliers(df: pd.DataFrame):
    df= df.query('minimum_nights < 20 &'
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
    # df.describe()
    return df


def matriz_corr(df: pd.DataFrame):
    # matriz de correlacao com o dataframe ja "limpo"
    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap='RdBu', fmt='.2f', square=True, linecolor='white', annot=True)
    plt.title('Matriz de Correlação')
    plt.show()








