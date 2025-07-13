import pandas as pd



# POR BAIRRO
'''
Fiz umas regressoes sem filtrar o bairro e usar a latitute e longitude que tem no arquivo
mas o R^2 tava ficando bem ruim (nao sei direito pq), mas acho que filtrar por bairro ta ok
cada bairro tem uma precificação diferente

Principais bairros em quantidade
Copacabana                  6645
Barra da Tijuca             2131
Ipanema                     1770
Recreio dos Bandeirantes    1361
Jacarepaguá                 1120
'''

from flask import Flask, render_template, request


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def estimate_price():
    if request.method == 'POST':
        # O que saiu do site
        neighbourhood = str(request.form.get('neighbourhood'))
        room_type = str(request.form.get('room_type'))
        accommodates = int(request.form.get('accommodates'))
        bathrooms = float(request.form.get('bathrooms'))
        bedrooms = int(request.form.get('bedrooms'))
        beds = int(request.form.get('beds'))
        minimum_nights = int(request.form.get('minimum_nights'))
        maximum_nights = int(request.form.get('maximum_nights'))
        availability_365 = int(request.form.get('availability_365'))
        host_is_superhost = 1 if request.form.get('host_is_superhost') == 'on' else 0
        number_of_reviews = int(request.form.get('number_of_reviews'))
        reviews_per_month = float(request.form.get('reviews_per_month'))
        review_scores_rating = float(request.form.get('review_scores_rating'))
        review_scores_accuracy = float(request.form.get('review_scores_accuracy'))
        review_scores_cleanliness = float(request.form.get('review_scores_cleanliness'))
        review_scores_checkin = float(request.form.get('review_scores_checkin'))
        review_scores_communication = float(request.form.get('review_scores_communication'))
        review_scores_location = float(request.form.get('review_scores_location'))
        review_scores_value = float(request.form.get('review_scores_value'))



        bairro = neighbourhood
        tipo_ap = room_type

        ap = {
            'is_superhost':        host_is_superhost,
            'accommodates':        accommodates,
            'bathrooms':           bathrooms,
            'bedrooms':            bedrooms,
            'beds':                beds,
            'minimum_nights':      minimum_nights,
            'maximum_nights':      maximum_nights,
            'number_of_reviews':   number_of_reviews,
            'reviews_per_month':   reviews_per_month,
            'review_scores_rating':        review_scores_rating,
            'review_scores_accuracy':      review_scores_accuracy,
            'review_scores_cleanliness':   review_scores_cleanliness,
            'review_scores_checkin':       review_scores_checkin,
            'review_scores_communication': review_scores_communication,
            'review_scores_location':      review_scores_location,
            'review_scores_value':         review_scores_value,
            'availability_365':    availability_365,
        }


        from exame_reg_com_tuning import model5, min_max_scaler
        cols_ap = list(ap.keys())
        df_ap = pd.DataFrame([ap], columns=cols_ap)
        X_novo = min_max_scaler.transform(df_ap) # normalizando igual

        preco_ap_rf = model5.predict(X_novo)[0] 

        predicted_price =  preco_ap_rf

        print(predicted_price)
        return render_template("index.html", predicted_price=predicted_price)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
