import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('./datasets/listings_rio_1.csv')


df['price_float'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
map_superhost = {'t': 1, 'f': 0}
df['is_superhost'] = df['host_is_superhost'].map(map_superhost)

df0 = df #salvando o df para plotar o grafico do mapa de calor

df = df[df['room_type'] == 'Entire home/apt']
df = df[df['neighbourhood_cleansed'] == 'Copacabana']


cols = [
    'price_float',
    'is_superhost',
    # 'latitude',
    # 'longitude',
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

df_clean = df[cols].dropna()

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
