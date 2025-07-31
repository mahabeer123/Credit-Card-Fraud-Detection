#!/usr/bin/env python3
"""
Standalone model training script for Streamlit deployment
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.distance import great_circle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def save_models():
    """Train and save models for deployment"""
    print("🔄 Loading data and training models...")
    
    # Create synthetic data for training
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic transaction data
    data = {
        'cc_num': np.random.randint(1000000000000000, 9999999999999999, n_samples),
        'amt': np.random.exponential(50, n_samples),
        'zip': np.random.randint(10000, 99999, n_samples),
        'lat': np.random.uniform(25, 50, n_samples),
        'long': np.random.uniform(-125, -65, n_samples),
        'city_pop': np.random.randint(1000, 1000000, n_samples),
        'unix_time': np.random.randint(1600000000, 1700000000, n_samples),
        'merch_lat': np.random.uniform(25, 50, n_samples),
        'merch_long': np.random.uniform(-125, -65, n_samples),
        'trans_hour': np.random.randint(0, 24, n_samples),
        'trans_day_of_week': np.random.randint(0, 7, n_samples),
        'trans_month': np.random.randint(1, 13, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.995, 0.005])
    }
    
    data = pd.DataFrame(data)
    
    # Calculate distance feature
    print("🔍 Calculating distance feature...")
    data['distance'] = data.apply(
        lambda row: great_circle(
            (row['lat'], row['long']), 
            (row['merch_lat'], row['merch_long'])
        ).miles, axis=1
    )
    
    # Prepare features
    feature_columns = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 
                      'unix_time', 'merch_lat', 'merch_long', 'trans_hour', 
                      'trans_day_of_week', 'trans_month', 'age', 'distance']
    
    X = data[feature_columns]
    y = data['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Manual class balancing
    fraud_indices = y_train[y_train == 1].index
    non_fraud_indices = y_train[y_train == 0].index
    target_size = min(len(fraud_indices), len(non_fraud_indices))
    
    fraud_sample = fraud_indices.to_series().sample(n=target_size, random_state=42).index
    non_fraud_sample = non_fraud_indices.to_series().sample(n=target_size, random_state=42).index
    balanced_indices = fraud_sample.union(non_fraud_sample)
    
    X_train_balanced = X_train.loc[balanced_indices]
    y_train_balanced = y_train.loc[balanced_indices]
    
    print("🤖 Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        n_jobs=-1,
        max_depth=15, 
        min_samples_split=20, 
        min_samples_leaf=10,
        class_weight='balanced'
    )
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
    
    print("✅ Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    print("💾 Saving models...")
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Models saved successfully!")
    
    return rf_model, scaler

if __name__ == "__main__":
    save_models() 