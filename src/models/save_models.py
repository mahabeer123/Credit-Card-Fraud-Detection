#!/usr/bin/env python3
"""
Save trained models from the comprehensive pipeline
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle
import warnings
warnings.filterwarnings('ignore')

def save_models():
    """Save the best model and scaler for the Streamlit app"""
    
    print("🔄 Loading data and training models...")
    
    # Load data
    fraud_train = pd.read_csv('../../data/fraudTrain.csv')
    fraud_test = pd.read_csv('../../data/fraudTest.csv')
    
    # Combine datasets
    data = pd.concat([fraud_train, fraud_test]).reset_index()
    data.drop(data.columns[:2], axis=1, inplace=True)
    
    # Sample for faster processing
    data = data.sample(n=100000, random_state=42).reset_index(drop=True)
    
    # Data cleaning
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['dob'] = pd.to_datetime(data['dob'])
    
    # Create derived features
    data['trans_hour'] = data['trans_date_trans_time'].dt.hour
    data['trans_day_of_week'] = data['trans_date_trans_time'].dt.day_name()
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['age'] = np.round((data['trans_date_trans_time'] - data['dob'])/np.timedelta64(1, 'Y'))
    
    # Calculate distance feature
    data['distance'] = data.apply(lambda row: great_circle((row['lat'], row['long']), 
                                             (row['merch_lat'], row['merch_long'])).kilometers, axis=1)
    
    # Drop unnecessary columns
    columns_to_drop = ['trans_date_trans_time', 'merchant', 'category', 'gender', 
                      'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
    data.drop(columns_to_drop, axis=1, inplace=True)
    
    # Handle categorical variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['trans_day_of_week'] = le.fit_transform(data['trans_day_of_week'])
    
    # Remove extreme outliers
    Q1 = data['amt'].quantile(0.25)
    Q3 = data['amt'].quantile(0.75)
    IQR = Q3 - Q1
    extreme_outlier_mask = (data['amt'] < (Q1 - 3 * IQR)) | (data['amt'] > (Q3 + 3 * IQR))
    data = data[~extreme_outlier_mask]
    
    # Prepare features and target
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balance classes manually
    fraud_indices = y_train[y_train == 1].index
    non_fraud_indices = y_train[y_train == 0].index
    
    n_fraud = len(fraud_indices)
    n_non_fraud = len(non_fraud_indices)
    target_size = min(n_fraud, n_non_fraud)
    
    fraud_sample = fraud_indices.to_series().sample(n=target_size, random_state=42).index
    non_fraud_sample = non_fraud_indices.to_series().sample(n=target_size, random_state=42).index
    balanced_indices = fraud_sample.union(non_fraud_sample)
    
    X_train_balanced = X_train.loc[balanced_indices]
    y_train_balanced = y_train.loc[balanced_indices]
    
    # Train Random Forest (best model)
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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    # Save models
    print("💾 Saving models...")
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("✅ Models saved successfully!")
    print("📁 Files created:")
    print("   - random_forest_model.pkl")
    print("   - scaler.pkl")
    
    return rf_model, scaler

if __name__ == "__main__":
    save_models() 