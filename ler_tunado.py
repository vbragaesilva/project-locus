import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_dataframe(path:str):
    return pd.read_csv(path)

def clean_dataframe(df: pd.DataFrame):
    clean_df = df.copy()

    # remover simbolos
    clean_df['price_float'] = clean_df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    # substituir t/f por 1/0
    map_superhost = {'t': 1, 'f': 0}
    clean_df['is_superhost'] = clean_df['host_is_superhost'].map(map_superhost)

    # remover N/A
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
    clean_df[cols].dropna()

    return clean_df

def filter_by_room_type(df: pd.DataFrame, room_type:str):
    # nosso interesse eh room_type='Entire home/apt'
    return df[df['room_type'] == room_type]

def filter_by_neighborhood(df: pd.DataFrame, bairro:str):
    return df[df['neighbourhood_cleansed'] == bairro]

def mapa_de_calor():
    pass

def 