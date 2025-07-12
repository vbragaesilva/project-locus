import os
import logging
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "airbnb-price-calculator-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    """Preprocess the dataframe for price prediction"""
    try:
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())
        
        # Handle categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown')
        
        # Convert boolean columns
        bool_columns = ['host_is_superhost']
        for col in bool_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0)
        
        return df_processed
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        return None

def build_prediction_model(df):
    """Build and train a price prediction model"""
    try:
        # Define feature columns that commonly exist in Airbnb datasets
        feature_columns = [
            'accommodates', 'bathrooms', 'bedrooms', 'beds',
            'minimum_nights', 'maximum_nights', 'number_of_reviews',
            'reviews_per_month', 'review_scores_rating', 'review_scores_accuracy',
            'review_scores_cleanliness', 'review_scores_checkin',
            'review_scores_communication', 'review_scores_location',
            'review_scores_value', 'availability_365', 'host_is_superhost'
        ]
        
        # Filter columns that exist in the dataset
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) == 0:
            logging.error("No matching feature columns found in dataset")
            return None, None, None
        
        # Prepare features and target
        X = df[available_features].copy()
        
        # Handle price column (target variable)
        if 'price' in df.columns:
            y = df['price'].copy()
            # Clean price column if it contains currency symbols
            if y.dtype == 'object':
                y = y.astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
                y = pd.to_numeric(y, errors='coerce')
        else:
            logging.error("Price column not found in dataset")
            return None, None, None
        
        # Remove rows with missing target values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            logging.error("No valid data rows after cleaning")
            return None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"Model trained successfully. MAE: {mae:.2f}, R²: {r2:.3f}")
        
        return model, scaler, available_features
    
    except Exception as e:
        logging.error(f"Error building prediction model: {str(e)}")
        return None, None, None

def predict_price(model, scaler, feature_columns, input_data):
    """Predict price based on input features"""
    try:
        # Create feature vector
        features = []
        for col in feature_columns:
            if col in input_data:
                features.append(input_data[col])
            else:
                features.append(0)  # Default value for missing features
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative price
    
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'datafile' not in request.files:
                flash('No file uploaded', 'error')
                return redirect(request.url)
            
            file = request.files['datafile']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Load and preprocess data
                df = pd.read_csv(filepath)
                df_processed = preprocess_data(df)
                
                if df_processed is None:
                    flash('Error processing the uploaded file', 'error')
                    return redirect(request.url)
                
                # Build model
                model, scaler, feature_columns = build_prediction_model(df_processed)
                
                if model is None:
                    flash('Error building prediction model. Please check your data format.', 'error')
                    return redirect(request.url)
                
                # Get form data
                input_data = {
                    'accommodates': float(request.form.get('accommodates', 0)),
                    'bathrooms': float(request.form.get('bathrooms', 0)),
                    'bedrooms': float(request.form.get('bedrooms', 0)),
                    'beds': float(request.form.get('beds', 0)),
                    'minimum_nights': float(request.form.get('minimum_nights', 0)),
                    'maximum_nights': float(request.form.get('maximum_nights', 0)),
                    'number_of_reviews': float(request.form.get('number_of_reviews', 0)),
                    'reviews_per_month': float(request.form.get('reviews_per_month', 0)),
                    'review_scores_rating': float(request.form.get('review_scores_rating', 0)),
                    'review_scores_accuracy': float(request.form.get('review_scores_accuracy', 0)),
                    'review_scores_cleanliness': float(request.form.get('review_scores_cleanliness', 0)),
                    'review_scores_checkin': float(request.form.get('review_scores_checkin', 0)),
                    'review_scores_communication': float(request.form.get('review_scores_communication', 0)),
                    'review_scores_location': float(request.form.get('review_scores_location', 0)),
                    'review_scores_value': float(request.form.get('review_scores_value', 0)),
                    'availability_365': float(request.form.get('availability_365', 0)),
                    'host_is_superhost': 1 if request.form.get('host_is_superhost') == 'on' else 0
                }
                
                # Make prediction
                predicted_price = predict_price(model, scaler, feature_columns, input_data)
                
                if predicted_price is None:
                    flash('Error making price prediction', 'error')
                    return redirect(request.url)
                
                # Clean up uploaded file
                os.remove(filepath)
                
                flash(f'Estimated Price: ${predicted_price:.2f}', 'success')
                return render_template('index.html', predicted_price=predicted_price)
            
            else:
                flash('Invalid file type. Please upload a CSV file.', 'error')
                return redirect(request.url)
        
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            flash('An error occurred while processing your request. Please try again.', 'error')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
