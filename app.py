from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Global variables to store loaded models
model = None
scaler = None
ohe = None
label_encoder = None

def load_models():
    """Load all the trained models and preprocessors"""
    global model, scaler, ohe, label_encoder
    
    try:
        # Load the trained model
        model = load_model('customer_churn_model.h5')
        print("✅ Model loaded successfully")
        
        # Load the scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully")
        
        # Load the one-hot encoder for geography
        with open('onehot_encoder_geography.pkl', 'rb') as f:
            ohe = pickle.load(f)
        print("✅ Geography encoder loaded successfully")
        
        # Load the label encoder for gender
        with open('label_encoder_gender.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Gender encoder loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Create DataFrame from input data
        df = pd.DataFrame([data])
        
        # Apply label encoding to Gender
        df['Gender'] = label_encoder.transform(df['Gender'])
        
        # Apply one-hot encoding to Geography
        geography_encoded = ohe.transform(df[['Geography']]).toarray()
        geography_feature_names = ohe.get_feature_names_out(['Geography'])
        
        # Create DataFrame with geography features
        geography_df = pd.DataFrame(geography_encoded, columns=geography_feature_names)
        
        # Drop original Geography column and concatenate encoded features
        df = df.drop('Geography', axis=1)
        df = pd.concat([df.reset_index(drop=True), geography_df.reset_index(drop=True)], axis=1)
        
        # Ensure columns are in the exact same order as training data
        expected_columns = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France',
            'Geography_Germany', 'Geography_Spain'
        ]
        
        # Reorder columns to match training data
        df = df[expected_columns]
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        return scaled_data
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

@app.route('/')
def index():
    """Main prediction page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                          'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert numeric fields to appropriate types
        try:
            data['CreditScore'] = int(data['CreditScore'])
            data['Age'] = int(data['Age'])
            data['Tenure'] = int(data['Tenure'])
            data['Balance'] = float(data['Balance'])
            data['NumOfProducts'] = int(data['NumOfProducts'])
            data['HasCrCard'] = int(data['HasCrCard'])
            data['IsActiveMember'] = int(data['IsActiveMember'])
            data['EstimatedSalary'] = float(data['EstimatedSalary'])
        except ValueError as e:
            return jsonify({'error': f'Invalid data type: {str(e)}'}), 400
        
        # Preprocess the data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data, verbose=0)
        prediction_proba = float(prediction[0][0])
        prediction_binary = int(prediction_proba > 0.5)
        
        # Prepare response
        response = {
            'churn_probability': round(prediction_proba, 4),
            'will_churn': prediction_binary == 1,
            'risk_level': get_risk_level(prediction_proba),
            'confidence': round(max(prediction_proba, 1 - prediction_proba), 4)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_risk_level(probability):
    """Determine risk level based on churn probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': all([model is not None, scaler is not None, ohe is not None, label_encoder is not None])
    })

if __name__ == '__main__':
    print("🚀 Starting Customer Churn Prediction App...")
    
    # Load models on startup
    if load_models():
        print("🎉 All models loaded successfully!")
        print("🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("💥 Failed to load models. Please check if all model files exist.")
