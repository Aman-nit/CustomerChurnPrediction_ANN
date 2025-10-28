# 🧠 Customer Churn Prediction - AI Web App

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)](https://python.org/)

A deep learning web application that predicts customer churn using Neural Networks with **86.5% accuracy**.

---

## 📊 Model Performance

| Metric        | Score |
| ------------- | ----- |
| **Accuracy**  | 86.5% |
| **AUC-ROC**   | 0.864 |
| **Precision** | 78.2% |
| **Recall**    | 81.7% |

## 🏗️ Neural Network Architecture

```
Input Layer (12 features)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Key Specifications:**

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy
- **Parameters:** 2,945 trainable params
- **Training:** Early stopping with 80/20 split

---

## 🚀 Features

✅ Real-time churn prediction with probability scores  
✅ Risk level classification (Low/Medium/High)  
✅ Interactive web interface with responsive design  
✅ REST API for predictions  
✅ Automated data preprocessing pipeline

---

## 🛠️ Tech Stack

**Backend:** Flask, TensorFlow/Keras, Pandas, NumPy, Scikit-learn  
**Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5  
**Data Processing:** StandardScaler, OneHotEncoder, LabelEncoder

---

## 📁 Project Structure

```
├── app.py                          # Flask application
├── customer_churn_model.h5         # Trained model
├── modelTrained.ipynb              # Training notebook
├── prediction.ipynb                # Prediction examples
├── Churn_Modelling.csv             # Dataset
├── requirements.txt                # Dependencies
├── templates/
│   └── index.html                  # Main web interface
└── static/
    ├── style.css                   # Styles
    └── app.js                      # Frontend logic
```

---

## 🚦 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Aman-nit/CustomerChurnPrediction_ANN.git
cd CustomerChurnPrediction_ANN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python app.py
```

### 4. Open Browser

Navigate to `http://localhost:5000`

---

## 📝 Input Features

| Feature          | Type        | Description            |
| ---------------- | ----------- | ---------------------- |
| Credit Score     | Numeric     | 300-850                |
| Geography        | Categorical | France/Germany/Spain   |
| Gender           | Categorical | Male/Female            |
| Age              | Numeric     | 18-100                 |
| Tenure           | Numeric     | Years with bank (0-10) |
| Balance          | Numeric     | Account balance        |
| Num of Products  | Numeric     | 1-4 products           |
| Has Credit Card  | Binary      | 0/1                    |
| Is Active Member | Binary      | 0/1                    |
| Estimated Salary | Numeric     | Annual salary          |

---

## 🔧 API Usage

### Prediction Endpoint

**POST** `/predict`

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 50000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 60000
}
```

**Response:**

```json
{
  "churn_probability": 0.2345,
  "will_churn": false,
  "risk_level": "Low",
  "confidence": 0.7655
}
```

---

## 📊 Model Training

The model was trained on 10,000 customer records with:

- **Training Split:** 80% train, 20% test
- **Epochs:** 100 (with early stopping)
- **Batch Size:** 32
- **Validation:** Used for early stopping

---

## 🎯 Key Skills Demonstrated

- Deep Learning (TensorFlow/Keras)
- Web Development (Flask)
- Data Preprocessing & Feature Engineering
- Model Deployment & Production
- REST API Design
- Frontend Development
- Git Version Control

---

## 📦 Dependencies

```
tensorflow==2.12.0
flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
```

---

## 📄 License

MIT License - Feel free to use this project for learning and portfolio purposes.

---

## 👨‍💻 Author

**Aman Kumar**  
[GitHub](https://github.com/Aman-nit) | [LinkedIn](#)

---

## 🌟 Acknowledgments

- Dataset: Bank customer churn dataset
- Framework: TensorFlow & Flask
- UI: Bootstrap 5

---

**⭐ Star this repo if you find it helpful!**
