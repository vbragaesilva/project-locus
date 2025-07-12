import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging

class AirbnbPriceCalculator:
    def __init__(self, csv_file_path):
        """
        Initialize the price calculator with a CSV file containing Airbnb data.
        
        Args:
            csv_file_path (str): Path to the CSV file with Airbnb listings data
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.scaler = StandardScaler()
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the CSV data."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            logging.info(f"Loaded {len(self.df)} records from {self.csv_file_path}")
            
            # Clean column names (remove spaces, make lowercase)
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Standardize price column name
            price_columns = ['price', 'price_per_night', 'nightly_price', 'cost', 'rate']
            price_col = None
            for col in price_columns:
                if col in self.df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                raise ValueError("No price column found. Expected columns: " + ", ".join(price_columns))
            
            # Rename to standard 'price' column
            if price_col != 'price':
                self.df['price'] = self.df[price_col]
            
            # Clean price data (remove $ signs, convert to float)
            if self.df['price'].dtype == 'object':
                self.df['price'] = self.df['price'].astype(str).str.replace('$', '').str.replace(',', '')
                self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
            
            # Handle boolean columns
            bool_columns = ['host_is_superhost', 'superhost']
            for col in bool_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).str.lower().isin(['true', 't', 'yes', 'y', '1'])
            
            # Remove rows with missing price data
            self.df = self.df.dropna(subset=['price'])
            
            # Remove outliers (prices beyond reasonable range)
            self.df = self.df[(self.df['price'] >= 10) & (self.df['price'] <= 10000)]
            
            logging.info(f"Cleaned data: {len(self.df)} valid records")
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def get_feature_columns(self):
        """Get the list of feature columns that match the input property data."""
        feature_mapping = {
            'accommodates': ['accommodates', 'max_guests', 'guest_capacity'],
            'bathrooms': ['bathrooms', 'bathroom_count', 'baths'],
            'bedrooms': ['bedrooms', 'bedroom_count'],
            'beds': ['beds', 'bed_count'],
            'minimum_nights': ['minimum_nights', 'min_nights'],
            'maximum_nights': ['maximum_nights', 'max_nights'],
            'number_of_reviews': ['number_of_reviews', 'review_count', 'total_reviews'],
            'reviews_per_month': ['reviews_per_month', 'monthly_reviews'],
            'review_scores_rating': ['review_scores_rating', 'overall_rating', 'rating'],
            'review_scores_accuracy': ['review_scores_accuracy', 'accuracy_rating'],
            'review_scores_cleanliness': ['review_scores_cleanliness', 'cleanliness_rating'],
            'review_scores_checkin': ['review_scores_checkin', 'checkin_rating'],
            'review_scores_communication': ['review_scores_communication', 'communication_rating'],
            'review_scores_location': ['review_scores_location', 'location_rating'],
            'review_scores_value': ['review_scores_value', 'value_rating'],
            'availability_365': ['availability_365', 'availability', 'days_available'],
            'host_is_superhost': ['host_is_superhost', 'superhost']
        }
        
        available_features = {}
        for feature, possible_cols in feature_mapping.items():
            for col in possible_cols:
                if col in self.df.columns:
                    available_features[feature] = col
                    break
        
        return available_features
    
    def calculate_price(self, property_data):
        """
        Calculate estimated price for a property based on similar listings.
        
        Args:
            property_data (dict): Dictionary containing property characteristics
            
        Returns:
            float: Estimated price per night
        """
        try:
            feature_columns = self.get_feature_columns()
            
            if len(feature_columns) == 0:
                raise ValueError("No matching feature columns found in the dataset")
            
            # Create feature matrix for existing properties
            feature_matrix = []
            property_features = []
            
            for feature, col_name in feature_columns.items():
                if col_name in self.df.columns:
                    # Fill missing values with median for numeric columns
                    if self.df[col_name].dtype in ['int64', 'float64']:
                        self.df[col_name] = self.df[col_name].fillna(self.df[col_name].median())
                    else:
                        self.df[col_name] = self.df[col_name].fillna(False)
                    
                    feature_matrix.append(self.df[col_name].values)
                    property_features.append(property_data.get(feature, 0))
            
            if len(feature_matrix) == 0:
                raise ValueError("No valid features found for price calculation")
            
            # Convert to numpy arrays
            feature_matrix = np.array(feature_matrix).T
            property_features = np.array(property_features).reshape(1, -1)
            
            # Standardize features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            property_features_scaled = self.scaler.transform(property_features)
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(property_features_scaled, feature_matrix_scaled)[0]
            
            # Get top 20% most similar properties
            top_percentile = max(1, int(len(similarity_scores) * 0.2))
            top_indices = np.argsort(similarity_scores)[-top_percentile:]
            
            # Calculate weighted average price
            similar_prices = self.df.iloc[top_indices]['price'].values
            similar_weights = similarity_scores[top_indices]
            
            # Normalize weights
            similar_weights = similar_weights / similar_weights.sum()
            
            # Calculate weighted average
            estimated_price = np.average(similar_prices, weights=similar_weights)
            
            # Apply adjustment factors based on property characteristics
            adjustment_factor = 1.0
            
            # Superhost premium
            if property_data.get('host_is_superhost', False):
                adjustment_factor *= 1.1
            
            # High rating bonus
            avg_rating = np.mean([
                property_data.get('review_scores_rating', 0),
                property_data.get('review_scores_accuracy', 0),
                property_data.get('review_scores_cleanliness', 0),
                property_data.get('review_scores_checkin', 0),
                property_data.get('review_scores_communication', 0),
                property_data.get('review_scores_location', 0),
                property_data.get('review_scores_value', 0)
            ])
            
            if avg_rating > 4.5:
                adjustment_factor *= 1.05
            elif avg_rating < 3.5 and avg_rating > 0:
                adjustment_factor *= 0.95
            
            # Apply adjustment
            estimated_price *= adjustment_factor
            
            logging.info(f"Calculated price: ${estimated_price:.2f} using {len(similar_prices)} similar properties")
            
            return round(estimated_price, 2)
            
        except Exception as e:
            logging.error(f"Error calculating price: {str(e)}")
            raise ValueError(f"Price calculation failed: {str(e)}")
