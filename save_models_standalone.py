#!/usr/bin/env python3
"""
Standalone model training script for Streamlit deployment
Creates a model compatible with current scikit-learn versions
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def save_models():
    """Train and save models for deployment"""
    print("🔄 Loading data and training models...")
    
    # Generate synthetic data that matches the real data structure
    np.random.seed(42)
    n_samples = 50000  # Smaller sample for faster training
    
    # Generate realistic synthetic data
    data = {
        'cc_num': np.random.randint(1000000000000000, 9999999999999999, n_samples),
        'amt': np.random.exponential(100, n_samples) + 10,  # Exponential distribution for amounts
        'zip': np.random.randint(10000, 99999, n_samples),
        'lat': np.random.uniform(25, 50, n_samples),  # US latitude range
        'long': np.random.uniform(-125, -65, n_samples),  # US longitude range
        'city_pop': np.random.randint(1000, 1000000, n_samples),
        'unix_time': np.random.randint(1500000000, 1600000000, n_samples),
        'merch_lat': np.random.uniform(25, 50, n_samples),
        'merch_long': np.random.uniform(-125, -65, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud rate
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add temporal features
    df['trans_hour'] = np.random.randint(0, 24, n_samples)
    df['trans_day_of_week'] = np.random.randint(0, 7, n_samples)
    df['trans_month'] = np.random.randint(1, 13, n_samples)
    df['age'] = np.random.randint(18, 80, n_samples)
    
    # Calculate distance feature
    def calculate_distance(row):
        try:
            return great_circle(
                (row['lat'], row['long']),
                (row['merch_lat'], row['merch_long'])
            ).kilometers
        except:
            return 0.0
    
    print("📍 Calculating distance features...")
    df['distance'] = df.apply(calculate_distance, axis=1)
    
    # Prepare features
    feature_columns = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 
                      'unix_time', 'merch_lat', 'merch_long', 'trans_hour', 
                      'trans_day_of_week', 'trans_month', 'age', 'distance']
    
    X = df[feature_columns]
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balance classes manually (since SMOTE causes issues)
    print("⚖️ Balancing classes...")
    fraud_indices = np.where(y_train == 1)[0]
    non_fraud_indices = np.where(y_train == 0)[0]
    
    # Sample equal numbers of fraud and non-fraud
    target_size = min(len(fraud_indices), len(non_fraud_indices))
    
    fraud_sample = np.random.choice(fraud_indices, size=target_size, replace=False)
    non_fraud_sample = np.random.choice(non_fraud_indices, size=target_size, replace=False)
    
    balanced_indices = np.concatenate([fraud_sample, non_fraud_sample])
    np.random.shuffle(balanced_indices)
    
    X_train_balanced = X_train_scaled[balanced_indices]
    y_train_balanced = y_train.iloc[balanced_indices]
    
    # Train Random Forest with current scikit-learn compatible parameters
    print("🤖 Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"📊 Model Performance:")
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
    return rf_model, scaler

if __name__ == "__main__":
    save_models() 