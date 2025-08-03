import ler_tunado as lt

df = lt.load_dataframe('./datasets/listings_rio_1.csv')
df = lt.format_dataframe(df)

lt.mapa_de_calor(df)

''' 'Entire home/apt' ou 'Private room' '''
df = lt.filter_by_room_type(df, 'Entire home/apt')


''' Alguns bairros
Copacabana                  6645
Barra da Tijuca             2131
Ipanema                     1770
Recreio dos Bandeirantes    1361
Jacarepaguá                 1120 '''
# Por enquanto estamos filtrando por bairro, mas a ideia é treinar o modelo com todos o dataframe.
# isso envolve a issue #14
df = lt.filter_by_neighborhood(df, 'Copacabana')


clean_df = lt.clean_dataframe(df)

lt.histograma(clean_df)
clean_df = lt.remove_outliers(clean_df)
lt.matriz_corr(clean_df)

# para uso nas regressões
final_df = clean_df