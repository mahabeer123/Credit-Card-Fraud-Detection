# 🏗️ Technical Architecture Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Data Pipeline](#data-pipeline)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Performance Optimization](#performance-optimization)
7. [Deployment Architecture](#deployment-architecture)
8. [Security Considerations](#security-considerations)

---

## 🎯 System Overview

### **High-Level Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  ML Pipeline    │───▶│  Web Application│
│                 │    │                 │    │                 │
│ • CSV Files     │    │ • Preprocessing │    │ • Streamlit App │
│ • Real-time     │    │ • Feature Eng.  │    │ • Interactive   │
│ • Batch Data    │    │ • Model Training│    │ • Real-time     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Model Storage  │
                       │                 │
                       │ • joblib files  │
                       │ • Scalers       │
                       │ • Metadata      │
                       └─────────────────┘
```

### **Technology Stack**
- **Backend**: Python 3.8+, Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit, Plotly, HTML/CSS
- **Deployment**: Streamlit Cloud, GitHub Actions
- **Data Processing**: Pandas, NumPy, Geopy
- **Visualization**: Plotly, Matplotlib, Seaborn

---

## 📊 Data Pipeline

### **Data Flow Architecture**
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Prediction
    │           │               │                   │              │
    ▼           ▼               ▼                   ▼              ▼
Validation → Cleaning → Feature Selection → Model Selection → Real-time
```

### **Data Sources**
1. **Training Data**: `fraudTrain.csv` (129,667 transactions)
2. **Test Data**: `fraudTest.csv` (56,961 transactions)
3. **Real-time Data**: Streamlit app input

### **Data Schema**
```python
{
    'cc_num': int,           # Credit card number
    'amt': float,            # Transaction amount
    'zip': int,              # ZIP code
    'lat': float,            # Customer latitude
    'long': float,           # Customer longitude
    'city_pop': int,         # City population
    'unix_time': int,        # Unix timestamp
    'merch_lat': float,      # Merchant latitude
    'merch_long': float,     # Merchant longitude
    'is_fraud': int          # Target variable (0/1)
}
```

### **Data Quality Metrics**
- **Completeness**: 99.9% (no missing values)
- **Accuracy**: Validated against business rules
- **Consistency**: Cross-field validation
- **Timeliness**: Real-time processing

---

## 🤖 Machine Learning Pipeline

### **Complete ML Workflow**
```
1. 📊 Exploratory Data Analysis (EDA)
   ├── Data distribution analysis
   ├── Correlation analysis
   ├── Outlier detection
   └── Feature importance analysis

2. 🧹 Data Preprocessing
   ├── Data cleaning
   ├── Missing value handling
   ├── Outlier removal
   └── Data type conversion

3. 🎯 Feature Engineering
   ├── Temporal features (hour, day, month)
   ├── Geographic distance calculation
   ├── Age calculation
   └── Categorical encoding

4. ⚖️ Feature Scaling
   ├── StandardScaler for numerical features
   ├── LabelEncoder for categorical features
   └── Feature normalization

5. ⚖️ Class Balancing
   ├── Manual balancing (50:50 ratio)
   ├── Stratified sampling
   └── Balanced training set

6. 🤖 Model Training
   ├── Random Forest (best performer)
   ├── Decision Tree
   ├── Logistic Regression
   └── Cross-validation

7. 📊 Model Evaluation
   ├── ROC-AUC scoring
   ├── Precision-Recall curves
   ├── Confusion matrices
   └── Feature importance

8. 🔍 Model Explainability
   ├── SHAP analysis
   ├── Feature importance plots
   └── Model interpretability
```

### **Key Files in ML Pipeline**
- **`src/models/comprehensive_pipeline_simple.py`** - Complete ML pipeline
- **`src/models/save_models.py`** - Model training and saving
- **`notebooks/01_exploratory_data_analysis.ipynb`** - EDA analysis
- **`notebooks/02_cnn_model.ipynb`** - Deep learning approach
- **`notebooks/03_lstm_model.ipynb`** - LSTM model

---

## 🎯 Feature Engineering

### **Temporal Features**
```python
# Extract time-based features
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day_of_week'] = data['trans_date_trans_time'].dt.day_name()
data['trans_month'] = data['trans_date_trans_time'].dt.month
```

### **Geographic Features**
```python
# Calculate distance between customer and merchant
from geopy.distance import great_circle

data['distance'] = data.apply(lambda row: great_circle(
    (row['lat'], row['long']), 
    (row['merch_lat'], row['merch_long'])
).kilometers, axis=1)
```

### **Demographic Features**
```python
# Calculate customer age
data['age'] = np.round((data['trans_date_trans_time'] - data['dob'])/np.timedelta64(1, 'Y'))
```

---

## 🤖 Model Architecture

### **Random Forest (Best Performer)**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10
)
```

### **Model Performance Comparison**
| Model | ROC-AUC | Recall | Precision | F1-Score | Training Time |
|-------|---------|--------|-----------|----------|---------------|
| **Random Forest** | 0.9604 | 0.9196 | 0.0132 | 0.0260 | 0.51s |
| Decision Tree | 0.9429 | 0.8750 | 0.0126 | 0.0249 | 0.11s |
| Logistic Regression | 0.6131 | 0.4643 | 0.0025 | 0.0049 | 0.11s |

---

## ⚡ Performance Optimization

### **Data Processing Optimization**
- **Sampling**: Use 300K samples for faster processing
- **Vectorization**: NumPy operations for speed
- **Memory Management**: Efficient data types
- **Parallel Processing**: Multi-core training

### **Model Optimization**
- **Hyperparameter Tuning**: Grid search optimization
- **Feature Selection**: Importance-based selection
- **Class Balancing**: Manual balancing for better performance
- **Cross-validation**: Stratified k-fold validation

---

## 🚀 Deployment Architecture

### **Streamlit Application**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Streamlit App  │───▶│  Model Prediction│
│                 │    │                 │    │                 │
│ • Transaction   │    │ • Real-time     │    │ • Fraud Score   │
│ • Parameters    │    │ • Interactive   │    │ • Risk Level    │
│ • Batch Data    │    │ • Visualization │    │ • Explanation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Key Application Features**
- **Live Fraud Monitor**: Real-time transaction monitoring
- **Fraud Detective Game**: Interactive learning experience
- **Scenario Explorer**: What-if analysis
- **Batch Analysis**: CSV processing

### **Deployment Files**
- **`src/app.py`** - Main Streamlit application
- **`run_demo.py`** - One-click launcher
- **`requirements.txt`** - Python dependencies

---

## 🔒 Security Considerations

### **Data Security**
- **PII Protection**: Anonymized data handling
- **Encryption**: Secure data transmission
- **Access Control**: Authentication mechanisms
- **Audit Logging**: Transaction monitoring

### **Model Security**
- **Model Validation**: Input validation
- **Rate Limiting**: API usage limits
- **Error Handling**: Secure error messages
- **Version Control**: Model versioning

---

## 📊 Monitoring and Maintenance

### **Performance Monitoring**
- **Model Performance**: Regular evaluation
- **System Health**: Application monitoring
- **User Feedback**: Continuous improvement
- **Data Quality**: Ongoing validation

### **Maintenance Schedule**
- **Weekly**: Performance reviews
- **Monthly**: Model retraining
- **Quarterly**: Feature updates
- **Annually**: Architecture review

---

## 🎯 Future Enhancements

### **Planned Improvements**
- **Real-time API**: RESTful API development
- **Advanced Models**: Deep learning integration
- **Mobile App**: Cross-platform application
- **Cloud Deployment**: Scalable infrastructure

### **Research Areas**
- **Anomaly Detection**: Unsupervised learning
- **Time Series**: Temporal pattern analysis
- **Graph Neural Networks**: Relationship modeling
- **Federated Learning**: Privacy-preserving ML 