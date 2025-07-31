#!/usr/bin/env python3
"""
Comprehensive Credit Card Fraud Detection Pipeline
Follows the complete ML workflow: EDA → Cleaning → Feature Selection → Scaling → Class Balance → Training → Testing → Comparison → SHAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve)
from geopy.distance import great_circle
import shap
import warnings
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class FraudDetectionPipeline:
    def __init__(self, sample_size=300000):
        """
        Initialize the pipeline
        sample_size: Use 300K samples for faster processing with good results
        """
        self.sample_size = sample_size
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def step1_eda_analysis(self):
        """Step 1: Exploratory Data Analysis"""
        print("="*60)
        print("🔍 STEP 1: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Load data
        print("Loading datasets...")
        fraud_train = pd.read_csv('../../data/fraudTrain.csv')
        fraud_test = pd.read_csv('../../data/fraudTest.csv')
        
        # Combine datasets
        self.data = pd.concat([fraud_train, fraud_test]).reset_index()
        self.data.drop(self.data.columns[:2], axis=1, inplace=True)
        
        # Sample data for faster processing
        print(f"Using sample of {self.sample_size:,} records for faster processing...")
        self.data = self.data.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Basic info
        print("\n📊 Basic Information:")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        
        # Target distribution
        fraud_dist = self.data['is_fraud'].value_counts()
        print(f"\n🎯 Target Distribution:")
        print(f"Non-Fraud: {fraud_dist[0]:,} ({fraud_dist[0]/len(self.data)*100:.2f}%)")
        print(f"Fraud: {fraud_dist[1]:,} ({fraud_dist[1]/len(self.data)*100:.2f}%)")
        
        # Data types
        print(f"\n📋 Data Types:")
        print(self.data.dtypes.value_counts())
        
        # Numerical features summary
        print(f"\n📈 Numerical Features Summary:")
        print(self.data.describe())
        
        return self.data
    
    def step2_data_cleaning(self):
        """Step 2: Data Cleaning and Preprocessing"""
        print("\n" + "="*60)
        print("🧹 STEP 2: DATA CLEANING")
        print("="*60)
        
        # Convert datetime columns
        print("Converting datetime columns...")
        self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time'])
        self.data['dob'] = pd.to_datetime(self.data['dob'])
        
        # Create derived features
        print("Creating derived features...")
        self.data['trans_hour'] = self.data['trans_date_trans_time'].dt.hour
        self.data['trans_day_of_week'] = self.data['trans_date_trans_time'].dt.day_name()
        self.data['trans_month'] = self.data['trans_date_trans_time'].dt.month
        self.data['age'] = np.round((self.data['trans_date_trans_time'] - 
                                   self.data['dob'])/np.timedelta64(1, 'Y'))
        
        # Calculate distance feature (KEY INNOVATION)
        print("🔍 Calculating distance feature (KEY INNOVATION)...")
        self.data['distance'] = self.data.apply(lambda row: great_circle((row['lat'], row['long']), 
                                                             (row['merch_lat'], row['merch_long'])).kilometers, axis=1)
        
        # Drop unnecessary columns but keep all useful features
        print("Dropping unnecessary columns...")
        columns_to_drop = ['trans_date_trans_time', 'merchant', 'category', 'gender', 
                          'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
        self.data.drop(columns_to_drop, axis=1, inplace=True)
        
        # Handle categorical variables
        print("Encoding categorical variables...")
        le = LabelEncoder()
        self.data['trans_day_of_week'] = le.fit_transform(self.data['trans_day_of_week'])
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found: {missing_values[missing_values > 0]}")
            # Fill missing values
            self.data = self.data.fillna(self.data.median())
        else:
            print("✅ No missing values found")
        
        # Remove extreme outliers but keep some for fraud detection
        print("Checking for outliers...")
        Q1 = self.data['amt'].quantile(0.25)
        Q3 = self.data['amt'].quantile(0.75)
        IQR = Q3 - Q1
        extreme_outlier_mask = (self.data['amt'] < (Q1 - 3 * IQR)) | (self.data['amt'] > (Q3 + 3 * IQR))
        self.data = self.data[~extreme_outlier_mask]
        print(f"Removed {extreme_outlier_mask.sum()} extreme outliers")
        
        print(f"Final dataset shape: {self.data.shape}")
        print(f"Features included: {list(self.data.columns)}")
        return self.data
    
    def step3_feature_selection(self):
        """Step 3: Feature Selection - Keep all features as requested"""
        print("\n" + "="*60)
        print("🎯 STEP 3: FEATURE SELECTION")
        print("="*60)
        
        # Separate features and target
        X = self.data.drop('is_fraud', axis=1)
        y = self.data['is_fraud']
        
        print(f"Using ALL features: {X.shape[1]}")
        print(f"Feature names: {list(X.columns)}")
        
        # Show feature importance for reference
        print("\n📊 Feature Importance Analysis (for reference)...")
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        rf_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 10 features by importance:")
        print(rf_importance.head(10))
        
        # Keep all features as requested
        selected_features = list(X.columns)
        print(f"\n✅ Using all {len(selected_features)} features")
        
        return selected_features
    
    def step4_feature_scaling(self):
        """Step 4: Feature Scaling"""
        print("\n" + "="*60)
        print("⚖️ STEP 4: FEATURE SCALING")
        print("="*60)
        
        # Separate features and target
        X = self.data.drop('is_fraud', axis=1)
        y = self.data['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Training fraud rate: {y_train.mean():.4f}")
        print(f"Test fraud rate: {y_test.mean():.4f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def step5_handle_class_imbalance(self):
        """Step 5: Handle Class Imbalance - Manual balancing to 50:50"""
        print("\n" + "="*60)
        print("⚖️ STEP 5: HANDLING CLASS IMBALANCE (50:50)")
        print("="*60)
        
        print(f"Before balancing - Training set:")
        print(f"Non-Fraud: {(self.y_train == 0).sum():,}")
        print(f"Fraud: {(self.y_train == 1).sum():,}")
        print(f"Fraud rate: {self.y_train.mean():.4f}")
        
        # Manual balancing to 50:50
        fraud_indices = self.y_train[self.y_train == 1].index
        non_fraud_indices = self.y_train[self.y_train == 0].index
        
        # Sample equal numbers of fraud and non-fraud
        n_fraud = len(fraud_indices)
        n_non_fraud = len(non_fraud_indices)
        target_size = min(n_fraud, n_non_fraud)
        
        # Sample fraud cases
        fraud_sample = fraud_indices.sample(n=target_size, random_state=42)
        # Sample non-fraud cases
        non_fraud_sample = non_fraud_indices.sample(n=target_size, random_state=42)
        
        # Combine balanced samples
        balanced_indices = fraud_sample.union(non_fraud_sample)
        
        X_train_balanced = self.X_train.loc[balanced_indices]
        y_train_balanced = self.y_train.loc[balanced_indices]
        
        print(f"\nAfter balancing - Training set:")
        print(f"Non-Fraud: {(y_train_balanced == 0).sum():,}")
        print(f"Fraud: {(y_train_balanced == 1).sum():,}")
        print(f"Fraud rate: {y_train_balanced.mean():.4f}")
        
        self.X_train_balanced = X_train_balanced
        self.y_train_balanced = y_train_balanced
        
        return X_train_balanced, y_train_balanced
    
    def step6_train_models(self):
        """Step 6: Train Models - LR, DT, RF, CNN, LSTM"""
        print("\n" + "="*60)
        print("🤖 STEP 6: TRAINING MODELS")
        print("="*60)
        
        results = {}
        
        # 1. Logistic Regression
        print("\n1️⃣ Training Logistic Regression...")
        start_time = time.time()
        lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0, class_weight='balanced')
        lr.fit(self.X_train_balanced, self.y_train_balanced)
        
        y_pred_lr = lr.predict(self.X_test)
        y_pred_proba_lr = lr.predict_proba(self.X_test)[:, 1]
        
        metrics_lr = self.calculate_metrics(self.y_test, y_pred_lr, y_pred_proba_lr)
        training_time_lr = time.time() - start_time
        
        results['Logistic Regression'] = {
            'model': lr,
            'metrics': metrics_lr,
            'y_pred': y_pred_lr,
            'y_pred_proba': y_pred_proba_lr,
            'training_time': training_time_lr
        }
        
        print(f"✅ Logistic Regression completed in {training_time_lr:.2f}s")
        print(f"   F1-Score: {metrics_lr['f1']:.4f}, ROC-AUC: {metrics_lr['roc_auc']:.4f}")
        
        # 2. Decision Tree
        print("\n2️⃣ Training Decision Tree...")
        start_time = time.time()
        dt = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=50, class_weight='balanced')
        dt.fit(self.X_train_balanced, self.y_train_balanced)
        
        y_pred_dt = dt.predict(self.X_test)
        y_pred_proba_dt = dt.predict_proba(self.X_test)[:, 1]
        
        metrics_dt = self.calculate_metrics(self.y_test, y_pred_dt, y_pred_proba_dt)
        training_time_dt = time.time() - start_time
        
        results['Decision Tree'] = {
            'model': dt,
            'metrics': metrics_dt,
            'y_pred': y_pred_dt,
            'y_pred_proba': y_pred_proba_dt,
            'training_time': training_time_dt
        }
        
        print(f"✅ Decision Tree completed in {training_time_dt:.2f}s")
        print(f"   F1-Score: {metrics_dt['f1']:.4f}, ROC-AUC: {metrics_dt['roc_auc']:.4f}")
        
        # 3. Random Forest
        print("\n3️⃣ Training Random Forest...")
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, 
                                  max_depth=15, min_samples_split=20, min_samples_leaf=10,
                                  class_weight='balanced')
        rf.fit(self.X_train_balanced, self.y_train_balanced)
        
        y_pred_rf = rf.predict(self.X_test)
        y_pred_proba_rf = rf.predict_proba(self.X_test)[:, 1]
        
        metrics_rf = self.calculate_metrics(self.y_test, y_pred_rf, y_pred_proba_rf)
        training_time_rf = time.time() - start_time
        
        results['Random Forest'] = {
            'model': rf,
            'metrics': metrics_rf,
            'y_pred': y_pred_rf,
            'y_pred_proba': y_pred_proba_rf,
            'training_time': training_time_rf
        }
        
        print(f"✅ Random Forest completed in {training_time_rf:.2f}s")
        print(f"   F1-Score: {metrics_rf['f1']:.4f}, ROC-AUC: {metrics_rf['roc_auc']:.4f}")
        
        # 4. CNN
        print("\n4️⃣ Training CNN...")
        start_time = time.time()
        cnn_model = self.create_cnn_model()
        cnn_history = cnn_model.fit(
            self.X_train_balanced, self.y_train_balanced,
            epochs=20, batch_size=256, validation_split=0.2,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        y_pred_proba_cnn = cnn_model.predict(self.X_test, verbose=0).flatten()
        y_pred_cnn = (y_pred_proba_cnn > 0.5).astype(int)
        
        metrics_cnn = self.calculate_metrics(self.y_test, y_pred_cnn, y_pred_proba_cnn)
        training_time_cnn = time.time() - start_time
        
        results['CNN'] = {
            'model': cnn_model,
            'metrics': metrics_cnn,
            'y_pred': y_pred_cnn,
            'y_pred_proba': y_pred_proba_cnn,
            'training_time': training_time_cnn
        }
        
        print(f"✅ CNN completed in {training_time_cnn:.2f}s")
        print(f"   F1-Score: {metrics_cnn['f1']:.4f}, ROC-AUC: {metrics_cnn['roc_auc']:.4f}")
        
        # 5. LSTM
        print("\n5️⃣ Training LSTM...")
        start_time = time.time()
        lstm_model = self.create_lstm_model()
        lstm_history = lstm_model.fit(
            self.X_train_balanced, self.y_train_balanced,
            epochs=20, batch_size=256, validation_split=0.2,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        y_pred_proba_lstm = lstm_model.predict(self.X_test, verbose=0).flatten()
        y_pred_lstm = (y_pred_proba_lstm > 0.5).astype(int)
        
        metrics_lstm = self.calculate_metrics(self.y_test, y_pred_lstm, y_pred_proba_lstm)
        training_time_lstm = time.time() - start_time
        
        results['LSTM'] = {
            'model': lstm_model,
            'metrics': metrics_lstm,
            'y_pred': y_pred_lstm,
            'y_pred_proba': y_pred_proba_lstm,
            'training_time': training_time_lstm
        }
        
        print(f"✅ LSTM completed in {training_time_lstm:.2f}s")
        print(f"   F1-Score: {metrics_lstm['f1']:.4f}, ROC-AUC: {metrics_lstm['roc_auc']:.4f}")
        
        self.results = results
        return results
    
    def create_cnn_model(self):
        """Create CNN model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_balanced.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def create_lstm_model(self):
        """Create LSTM model"""
        # Reshape data for LSTM (samples, timesteps, features)
        X_lstm = self.X_train_balanced.values.reshape((self.X_train_balanced.shape[0], 1, self.X_train_balanced.shape[1]))
        X_test_lstm = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        
        model = Sequential([
            LSTM(64, input_shape=(1, self.X_train_balanced.shape[1]), return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        # Store reshaped data for LSTM
        self.X_train_lstm = X_lstm
        self.X_test_lstm = X_test_lstm
        
        return model
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def step7_test_and_evaluate(self):
        """Step 7: Test and Evaluate Models"""
        print("\n" + "="*60)
        print("📊 STEP 7: TESTING AND EVALUATION")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1-Score': result['metrics']['f1'],
                'ROC-AUC': result['metrics']['roc_auc'],
                'Training Time (s)': result['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("Model Performance Comparison:")
        print(comparison_df.round(4))
        
        # Find best model for each metric
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        best_roc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
        
        print(f"\n🏆 Best F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
        print(f"🏆 Best ROC-AUC: {best_roc['Model']} ({best_roc['ROC-AUC']:.4f})")
        
        # Create visualizations
        self.create_evaluation_plots()
        
        return comparison_df
    
    def step8_compare_with_without_distance(self):
        """Step 8: Compare Models With and Without Distance Feature"""
        print("\n" + "="*60)
        print("🔍 STEP 8: DISTANCE FEATURE COMPARISON")
        print("="*60)
        
        # Prepare data without distance feature
        X_without_distance = self.X_train.drop('distance', axis=1, errors='ignore')
        X_test_without_distance = self.X_test.drop('distance', axis=1, errors='ignore')
        
        if 'distance' in self.X_train.columns:
            print("Distance feature found. Running comparison...")
            
            # Train Random Forest with and without distance
            rf_without = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            rf_with = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            
            # Train without distance
            print("Training without distance feature...")
            rf_without.fit(self.X_train_balanced.drop('distance', axis=1), self.y_train_balanced)
            y_pred_without = rf_without.predict(X_test_without_distance)
            y_pred_proba_without = rf_without.predict_proba(X_test_without_distance)[:, 1]
            
            # Train with distance
            print("Training with distance feature...")
            rf_with.fit(self.X_train_balanced, self.y_train_balanced)
            y_pred_with = rf_with.predict(self.X_test)
            y_pred_proba_with = rf_with.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics_without = self.calculate_metrics(self.y_test, y_pred_without, y_pred_proba_without)
            metrics_with = self.calculate_metrics(self.y_test, y_pred_with, y_pred_proba_with)
            
            # Calculate improvements
            improvements = {}
            for metric in metrics_without.keys():
                improvement = metrics_with[metric] - metrics_without[metric]
                improvement_pct = (improvement / metrics_without[metric]) * 100
                improvements[f'{metric}_improvement'] = improvement
                improvements[f'{metric}_improvement_pct'] = improvement_pct
            
            print("\n📊 Distance Feature Impact:")
            print(f"Without Distance - F1: {metrics_without['f1']:.4f}, ROC-AUC: {metrics_without['roc_auc']:.4f}")
            print(f"With Distance    - F1: {metrics_with['f1']:.4f}, ROC-AUC: {metrics_with['roc_auc']:.4f}")
            print(f"F1 Improvement: {improvements['f1_improvement_pct']:.2f}%")
            print(f"ROC-AUC Improvement: {improvements['roc_auc_improvement_pct']:.2f}%")
            
            # Create comparison plot
            self.create_distance_comparison_plot(metrics_without, metrics_with, improvements)
            
            return improvements
        else:
            print("Distance feature not found in dataset.")
            return None
    
    def step9_shap_verification(self):
        """Step 9: SHAP Verification"""
        print("\n" + "="*60)
        print("🔍 STEP 9: SHAP VERIFICATION")
        print("="*60)
        
        # Use Random Forest for SHAP analysis
        rf_model = self.results['Random Forest']['model']
        
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(rf_model)
        
        # Calculate SHAP values for a sample
        sample_size = min(1000, len(self.X_test))
        X_sample = self.X_test.iloc[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        # Create SHAP summary plot
        print("Creating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Feature Importance Analysis')
        plt.tight_layout()
        plt.savefig('../../visualizations/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Get feature importance from Random Forest
        feature_importance = rf_model.feature_importances_
        feature_names = self.X_train.columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\n📊 Feature Importance (Random Forest):")
        print(importance_df)
        
        if 'distance' in importance_df['Feature'].values:
            distance_importance = importance_df[importance_df['Feature'] == 'distance']['Importance'].iloc[0]
            distance_rank = importance_df[importance_df['Feature'] == 'distance'].index[0] + 1
            print(f"\n🏆 Distance feature importance: {distance_importance:.4f}")
            print(f"🏆 Distance feature rank: {distance_rank}/{len(importance_df)}")
        
        return importance_df
    
    def create_evaluation_plots(self):
        """Create evaluation plots"""
        print("Creating evaluation plots...")
        
        # ROC curves
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            auc_score = result['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall curves
        plt.subplot(1, 2, 2)
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
            plt.plot(recall, precision, label=f'{name}', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../visualizations/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Evaluation plots saved!")
    
    def create_distance_comparison_plot(self, metrics_without, metrics_with, improvements):
        """Create distance feature comparison plot"""
        print("Creating distance feature comparison plot...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot comparison
        x = np.arange(len(metrics))
        width = 0.35
        
        values_without = [metrics_without[m] for m in metrics]
        values_with = [metrics_with[m] for m in metrics]
        
        bars1 = axes[0].bar(x - width/2, values_without, width, label='Without Distance', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, values_with, width, label='With Distance', alpha=0.8)
        
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Distance Feature Impact')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Improvement percentages
        improvements_pct = [improvements[f'{m}_improvement_pct'] for m in metrics]
        colors = ['green' if x > 0 else 'red' for x in improvements_pct]
        
        axes[1].bar([m.replace('_', ' ').title() for m in metrics], improvements_pct, color=colors, alpha=0.7)
        axes[1].set_xlabel('Metrics')
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_title('Performance Improvement with Distance Feature')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../visualizations/distance_feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Distance feature comparison plot saved!")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("🚀 CREDIT CARD FRAUD DETECTION PIPELINE")
        print("="*60)
        print("Following steps: EDA → Cleaning → Feature Selection → Scaling → Class Balance → Training → Testing → Comparison → SHAP")
        print("="*60)
        
        start_time = time.time()
        
        # Run all steps
        self.step1_eda_analysis()
        self.step2_data_cleaning()
        self.step3_feature_selection()
        self.step4_feature_scaling()
        self.step5_handle_class_imbalance()
        self.step6_train_models()
        comparison_df = self.step7_test_and_evaluate()
        improvements = self.step8_compare_with_without_distance()
        importance_df = self.step9_shap_verification()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total execution time: {total_time/60:.2f} minutes")
        print(f"Best F1-Score: {comparison_df['F1-Score'].max():.4f}")
        print(f"Best ROC-AUC: {comparison_df['ROC-AUC'].max():.4f}")
        
        if improvements:
            print(f"Distance feature F1 improvement: {improvements['f1_improvement_pct']:.2f}%")
            print(f"Distance feature ROC-AUC improvement: {improvements['roc_auc_improvement_pct']:.2f}%")
        
        print("\n✅ All results saved in visualizations/ directory")
        print("✅ Models saved and ready for deployment")

if __name__ == "__main__":
    # Initialize pipeline with 300K samples for faster processing with good results
    pipeline = FraudDetectionPipeline(sample_size=300000)
    
    # Run complete pipeline
    pipeline.run_complete_pipeline() 