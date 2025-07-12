# Airbnb Price Calculator

## Overview

This is a Flask-based web application that provides Airbnb property price estimation using machine learning models. The application allows users to upload CSV datasets containing Airbnb listing data and get price predictions based on property characteristics using similarity matching and regression models.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: HTML5, CSS3, JavaScript, Bootstrap
- **Styling**: Custom CSS with gradient backgrounds, modern card-based UI
- **Responsiveness**: Bootstrap grid system for mobile-first design
- **User Experience**: File upload validation, loading states, flash messages for feedback

### Backend Architecture
- **Framework**: Flask (Python)
- **Structure**: Modular design with separate components
  - `app.py`: Main Flask application with routes and request handling
  - `price_calculator.py`: Core price prediction logic and data processing
  - `main.py`: Application entry point for deployment
- **Data Processing**: Pandas for CSV handling, NumPy for numerical operations
- **Machine Learning**: Scikit-learn for regression models and preprocessing

## Key Components

### 1. Web Application (`app.py`)
- **Purpose**: Handles HTTP requests, file uploads, and web interface
- **Features**:
  - File upload with validation (CSV only, 16MB limit)
  - Flash messaging for user feedback
  - Request logging and debugging
  - Proxy fix for deployment compatibility

### 2. Price Calculator (`price_calculator.py`)
- **Purpose**: Core business logic for price prediction
- **Features**:
  - CSV data loading and preprocessing
  - Column name standardization
  - Data cleaning (price formatting, missing values)
  - Similarity-based price calculation using cosine similarity
  - Support for various price column naming conventions

### 3. Frontend Components
- **Template**: Single-page application with upload form
- **Styling**: Modern gradient design with glassmorphism effects
- **JavaScript**: Form validation, file upload handling, loading states

## Data Flow

1. **Upload Phase**:
   - User uploads CSV file through web interface
   - File validation (type, size) performed client-side and server-side
   - File saved to uploads directory with secure filename

2. **Processing Phase**:
   - CSV data loaded into pandas DataFrame
   - Column names standardized (lowercase, underscore-separated)
   - Price column identified and cleaned (remove currency symbols)
   - Missing values handled with median/mode imputation

3. **Prediction Phase**:
   - Feature extraction and preprocessing
   - Similarity calculation using cosine similarity
   - Price estimation based on similar properties
   - Results returned to user interface

## External Dependencies

### Python Libraries
- **Flask**: Web framework for HTTP handling
- **Pandas**: Data manipulation and CSV processing
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Werkzeug**: WSGI utilities and security

### Frontend Libraries
- **Bootstrap**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements

## Deployment Strategy

### Configuration
- **Environment Variables**: Session secret key configurable via environment
- **File Handling**: Secure upload directory with size limits
- **Proxy Support**: ProxyFix middleware for deployment behind reverse proxies

### Deployment Options
- **Development**: Direct Flask development server (`python main.py`)
- **Production**: WSGI-compatible (can be deployed with Gunicorn, uWSGI)
- **Platform**: Designed for cloud deployment (environment variable configuration)

### Security Considerations
- Secure filename handling for uploads
- File type and size validation
- Session key protection
- Input sanitization for data processing

## Notable Architectural Decisions

### 1. Modular Design
- **Problem**: Separation of concerns between web interface and business logic
- **Solution**: Separate `AirbnbPriceCalculator` class for core functionality
- **Benefits**: Easier testing, reusability, maintainability

### 2. Flexible Data Schema
- **Problem**: Airbnb datasets may have varying column names
- **Solution**: Multiple possible column name mappings for price and features
- **Benefits**: Works with different dataset formats without modification

### 3. Client-Side Validation
- **Problem**: Provide immediate feedback for file uploads
- **Solution**: JavaScript validation with server-side backup
- **Benefits**: Better user experience, reduced server load

### 4. Similarity-Based Pricing
- **Problem**: Need robust price estimation without extensive feature engineering
- **Solution**: Cosine similarity matching with existing listings
- **Benefits**: Handles diverse property types, requires minimal data preprocessing