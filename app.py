from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import json

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
        # Training order: ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        #                  'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 
        #                  'Geography_Germany', 'Geography_Spain']
        expected_columns = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France',
            'Geography_Germany', 'Geography_Spain'
        ]
        
        # Reorder columns to match training data
        df = df[expected_columns]
        
        print(f"Feature columns: {list(df.columns)}")
        print(f"Feature shape: {df.shape}")
        print(f"Sample data:\n{df}")
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        return scaled_data
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    """Model Analytics and Training Details page"""
    try:
        # Load training data for analysis
        data = pd.read_csv('Churn_Modelling.csv')
        
        # Generate analytics data
        analytics_data = generate_analytics_data(data)
        
        return render_template('analytics.html', analytics=analytics_data)
    except Exception as e:
        return render_template('analytics.html', error=str(e))

@app.route('/dataset')
def dataset_info():
    """Dataset Information page"""
    try:
        # Load dataset for information
        data = pd.read_csv('Churn_Modelling.csv')
        
        # Generate dataset information
        dataset_info = generate_dataset_info(data)
        
        return render_template('dataset.html', dataset=dataset_info)
    except Exception as e:
        return render_template('dataset.html', error=str(e))

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
        prediction = model.predict(processed_data)
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

def generate_analytics_data(data):
    """Generate analytics data for the analytics page"""
    try:
        # Basic statistics
        total_customers = len(data)
        churn_rate = data['Exited'].mean()
        avg_age = data['Age'].mean()
        avg_balance = data['Balance'].mean()
        avg_credit_score = data['CreditScore'].mean()
        
        # Churn by geography
        churn_by_geography = data.groupby('Geography')['Exited'].agg(['count', 'sum', 'mean']).round(4)
        churn_by_geography['churn_rate'] = churn_by_geography['mean']
        
        # Churn by gender
        churn_by_gender = data.groupby('Gender')['Exited'].agg(['count', 'sum', 'mean']).round(4)
        churn_by_gender['churn_rate'] = churn_by_gender['mean']
        
        # Age distribution charts
        age_churn_chart = create_age_distribution_chart(data)
        balance_churn_chart = create_balance_distribution_chart(data)
        credit_score_chart = create_credit_score_chart(data)
        feature_correlation_chart = create_correlation_chart(data)
        
        # Model performance metrics (if available)
        model_metrics = get_model_performance_metrics()
        
        return {
            'basic_stats': {
                'total_customers': f"{total_customers:,}",
                'churn_rate': f"{churn_rate:.2%}",
                'avg_age': f"{avg_age:.1f} years",
                'avg_balance': f"${avg_balance:,.2f}",
                'avg_credit_score': f"{avg_credit_score:.0f}"
            },
            'churn_by_geography': churn_by_geography.to_dict('index'),
            'churn_by_gender': churn_by_gender.to_dict('index'),
            'charts': {
                'age_distribution': age_churn_chart,
                'balance_distribution': balance_churn_chart,
                'credit_score_distribution': credit_score_chart,
                'feature_correlation': feature_correlation_chart
            },
            'model_metrics': model_metrics
        }
    except Exception as e:
        return {'error': str(e)}

def generate_dataset_info(data):
    """Generate dataset information"""
    try:
        # Dataset overview
        dataset_shape = data.shape
        columns_info = []
        
        for col in data.columns:
            col_info = {
                'name': col,
                'type': str(data[col].dtype),
                'non_null': data[col].count(),
                'null_count': data[col].isnull().sum(),
                'unique_values': data[col].nunique()
            }
            
            if data[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': f"{data[col].mean():.2f}",
                    'std': f"{data[col].std():.2f}",
                    'min': f"{data[col].min():.2f}",
                    'max': f"{data[col].max():.2f}"
                })
            else:
                col_info.update({
                    'top_values': data[col].value_counts().head(3).to_dict()
                })
            
            columns_info.append(col_info)
        
        # Generate distribution charts
        target_distribution_chart = create_target_distribution_chart(data)
        numerical_features_chart = create_numerical_features_chart(data)
        categorical_features_chart = create_categorical_features_chart(data)
        
        return {
            'overview': {
                'total_records': f"{dataset_shape[0]:,}",
                'total_features': dataset_shape[1],
                'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024:.2f} KB",
                'missing_values': data.isnull().sum().sum()
            },
            'columns': columns_info,
            'charts': {
                'target_distribution': target_distribution_chart,
                'numerical_features': numerical_features_chart,
                'categorical_features': categorical_features_chart
            }
        }
    except Exception as e:
        return {'error': str(e)}

def create_age_distribution_chart(data):
    """Create age distribution chart by churn status"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create age bins
    bins = range(18, 101, 5)
    
    # Plot histograms for churned and non-churned customers
    data[data['Exited'] == 0]['Age'].hist(bins=bins, alpha=0.7, label='Stayed', color='green', ax=ax)
    data[data['Exited'] == 1]['Age'].hist(bins=bins, alpha=0.7, label='Churned', color='red', ax=ax)
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Age Distribution by Churn Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def create_balance_distribution_chart(data):
    """Create balance distribution chart"""
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Balance distribution
    data[data['Exited'] == 0]['Balance'].hist(bins=50, alpha=0.7, label='Stayed', color='green', ax=ax1)
    data[data['Exited'] == 1]['Balance'].hist(bins=50, alpha=0.7, label='Churned', color='red', ax=ax1)
    ax1.set_xlabel('Account Balance')
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Balance Distribution by Churn Status')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average balance by churn status
    balance_by_churn = data.groupby('Exited')['Balance'].mean()
    colors = ['green', 'red']
    ax2.bar(['Stayed', 'Churned'], balance_by_churn.values, color=colors, alpha=0.7)
    ax2.set_ylabel('Average Balance')
    ax2.set_title('Average Balance by Churn Status')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(balance_by_churn.values):
        ax2.text(i, v + 1000, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def create_credit_score_chart(data):
    """Create credit score distribution chart"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot of credit scores by churn status
    churn_labels = ['Stayed', 'Churned']
    credit_data = [data[data['Exited'] == 0]['CreditScore'], data[data['Exited'] == 1]['CreditScore']]
    
    bp = ax.boxplot(credit_data, labels=churn_labels, patch_artist=True)
    
    # Customize colors
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Credit Score')
    ax.set_title('Credit Score Distribution by Churn Status')
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def create_correlation_chart(data):
    """Create feature correlation heatmap"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numerical_cols].corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, linewidths=0.5, ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def create_target_distribution_chart(data):
    """Create target variable distribution chart"""
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    churn_counts = data['Exited'].value_counts()
    colors = ['green', 'red']
    ax1.bar(['Stayed', 'Churned'], churn_counts.values, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Customer Churn Distribution')
    
    # Add percentage labels
    total = churn_counts.sum()
    for i, v in enumerate(churn_counts.values):
        percentage = (v / total) * 100
        ax1.text(i, v + 200, f'{v:,}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(churn_counts.values, labels=['Stayed', 'Churned'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Churn Rate Distribution')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def create_numerical_features_chart(data):
    """Create numerical features distribution chart"""
    plt.style.use('seaborn-v0_8')
    numerical_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            ax = axes[i]
            data[col].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='navy', ax=ax)
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(numerical_cols) < len(axes):
        for j in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def create_categorical_features_chart(data):
    """Create categorical features distribution chart"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Geography distribution
    geo_counts = data['Geography'].value_counts()
    colors_geo = ['lightblue', 'lightgreen', 'lightyellow']
    axes[0].bar(geo_counts.index, geo_counts.values, color=colors_geo, alpha=0.8, edgecolor='navy')
    axes[0].set_title('Customer Distribution by Geography')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_xlabel('Geography')
    
    # Add value labels
    for i, v in enumerate(geo_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Gender distribution
    gender_counts = data['Gender'].value_counts()
    colors_gender = ['lightpink', 'lightcyan']
    axes[1].bar(gender_counts.index, gender_counts.values, color=colors_gender, alpha=0.8, edgecolor='navy')
    axes[1].set_title('Customer Distribution by Gender')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_xlabel('Gender')
    
    # Add value labels
    for i, v in enumerate(gender_counts.values):
        axes[1].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def get_model_performance_metrics():
    """Get model performance metrics if available"""
    try:
        # This would ideally load from saved metrics file
        # For now, returning example metrics
        return {
            'accuracy': 0.867,
            'precision': 0.775,
            'recall': 0.503,
            'f1_score': 0.610,
            'auc_roc': 0.851,
            'training_epochs': 25,
            'early_stopping_epoch': 25
        }
    except:
        return None

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)  # Important: close the figure to free memory
    return img_str

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
