import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           confusion_matrix, accuracy_score, roc_auc_score,
                           roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from geopy.distance import great_circle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and preprocess the credit card fraud data"""
    print("Loading data...")
    
    # Load datasets
    fraud_train = pd.read_csv('../../data/fraudTrain.csv')
    fraud_test = pd.read_csv('../../data/fraudTest.csv')
    
    # Concatenate datasets
    data = pd.concat([fraud_train, fraud_test]).reset_index()
    data.drop(data.columns[:2], axis=1, inplace=True)
    
    # Drop unnecessary columns
    data.drop(['trans_date_trans_time', 'merchant', 'category', 'gender', 
               'first', 'last', 'street', 'city', 'state', 
               'job', 'dob', 'trans_num'], axis=1, inplace=True)
    
    # Calculate distance feature - THE KEY INNOVATION
    print("🔍 Calculating distance feature (KEY INNOVATION)...")
    data['distance'] = data.apply(lambda row: great_circle((row['lat'], row['long']), 
                                                             (row['merch_lat'], row['merch_long'])).kilometers, axis=1)
    
    # Select features
    features_without_distance = ['zip', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time']
    features_with_distance = ['zip', 'lat', 'long', 'merch_lat', 'merch_long', 'unix_time', 'distance']
    X_without = data[features_without_distance]
    X_with = data[features_with_distance]
    y = data['is_fraud']
    
    print(f"Dataset shape: {X_with.shape}")
    print(f"Fraud rate: {y.mean():.4f}")
    print(f"Distance feature statistics:")
    print(data['distance'].describe())
    
    return X_without, X_with, y

def train_models_with_comparison(X_without, X_with, y):
    """Train models with and without distance feature for comparison"""
    print("\n🏆 Training models to demonstrate distance feature impact...")
    
    # Preprocess data
    scaler_without = StandardScaler()
    scaler_with = StandardScaler()
    
    # Split data
    X_train_without, X_test_without, y_train, y_test = train_test_split(
        X_without, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_with, X_test_with, _, _ = train_test_split(
        X_with, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_without_scaled = scaler_without.fit_transform(X_train_without)
    X_test_without_scaled = scaler_without.transform(X_test_without)
    X_train_with_scaled = scaler_with.fit_transform(X_train_with)
    X_test_with_scaled = scaler_with.transform(X_test_with)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train without distance
        model_without = model.__class__(**model.get_params())
        model_without.fit(X_train_without_scaled, y_train)
        y_pred_without = model_without.predict(X_test_without_scaled)
        y_pred_proba_without = model_without.predict_proba(X_test_without_scaled)[:, 1]
        
        # Train with distance
        model_with = model.__class__(**model.get_params())
        model_with.fit(X_train_with_scaled, y_train)
        y_pred_with = model_with.predict(X_test_with_scaled)
        y_pred_proba_with = model_with.predict_proba(X_test_with_scaled)[:, 1]
        
        # Calculate metrics
        metrics_without = {
            'accuracy': accuracy_score(y_test, y_pred_without),
            'precision': precision_score(y_test, y_pred_without),
            'recall': recall_score(y_test, y_pred_without),
            'f1': f1_score(y_test, y_pred_without),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_without)
        }
        
        metrics_with = {
            'accuracy': accuracy_score(y_test, y_pred_with),
            'precision': precision_score(y_test, y_pred_with),
            'recall': recall_score(y_test, y_pred_with),
            'f1': f1_score(y_test, y_pred_with),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_with)
        }
        
        # Calculate improvements
        improvements = {}
        for metric in metrics_without.keys():
            improvement = metrics_with[metric] - metrics_without[metric]
            improvement_pct = (improvement / metrics_without[metric]) * 100
            improvements[f'{metric}_improvement'] = improvement
            improvements[f'{metric}_improvement_pct'] = improvement_pct
        
        results[name] = {
            'model_without': model_without,
            'model_with': model_with,
            'scaler_without': scaler_without,
            'scaler_with': scaler_with,
            'metrics_without': metrics_without,
            'metrics_with': metrics_with,
            'improvements': improvements,
            'y_test': y_test,
            'y_pred_without': y_pred_without,
            'y_pred_with': y_pred_with,
            'y_pred_proba_without': y_pred_proba_without,
            'y_pred_proba_with': y_pred_proba_with
        }
        
        print(f"{name} Results:")
        print(f"  Without Distance - F1: {metrics_without['f1']:.4f}, ROC-AUC: {metrics_without['roc_auc']:.4f}")
        print(f"  With Distance    - F1: {metrics_with['f1']:.4f}, ROC-AUC: {metrics_with['roc_auc']:.4f}")
        print(f"  F1 Improvement: {improvements['f1_improvement_pct']:.2f}%")
        print(f"  ROC-AUC Improvement: {improvements['roc_auc_improvement_pct']:.2f}%")
    
    return results

def create_innovation_comparison_plot(results):
    """Create a comparison plot highlighting the distance feature innovation"""
    print("\n📊 Creating innovation comparison plot...")
    
    # Prepare data for plotting
    model_names = list(results.keys())
    metrics = ['f1', 'roc_auc', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values_without = [results[name]['metrics_without'][metric] for name in model_names]
        values_with = [results[name]['metrics_with'][metric] for name in model_names]
        improvements = [results[name]['improvements'][f'{metric}_improvement_pct'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = axes[i].bar(x - width/2, values_without, width, label='Without Distance', alpha=0.8)
        bars2 = axes[i].bar(x + width/2, values_with, width, label='With Distance', alpha=0.8)
        
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add improvement percentages
        for j, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
            axes[i].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                        f'+{improvement:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('../../visualizations/distance_feature_innovation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Innovation comparison plot saved!")

def create_roc_curves_comparison(results):
    """Create ROC curves comparing with and without distance feature"""
    print("Creating ROC curves comparison...")
    
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        # Without distance
        fpr_without, tpr_without, _ = roc_curve(result['y_test'], result['y_pred_proba_without'])
        auc_without = result['metrics_without']['roc_auc']
        
        # With distance
        fpr_with, tpr_with, _ = roc_curve(result['y_test'], result['y_pred_proba_with'])
        auc_with = result['metrics_with']['roc_auc']
        
        plt.plot(fpr_without, tpr_without, '--', alpha=0.7, 
                label=f'{name} Without Distance (AUC = {auc_without:.3f})')
        plt.plot(fpr_with, tpr_with, '-', linewidth=2,
                label=f'{name} With Distance (AUC = {auc_with:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k:', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Distance Feature Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../../visualizations/roc_curves_distance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ROC curves comparison saved!")

def save_models_and_results(results):
    """Save trained models and results"""
    print("\nSaving models and results...")
    
    # Save models with distance (primary models)
    for name, result in results.items():
        joblib.dump(result['model_with'], f'../../models/{name.lower().replace(" ", "_")}_model.pkl')
        joblib.dump(result['scaler_with'], f'../../models/{name.lower().replace(" ", "_")}_scaler.pkl')
    
    # Save comparison results
    comparison_summary = []
    for name, result in results.items():
        comparison_summary.append({
            'Model': name,
            'F1_Without_Distance': result['metrics_without']['f1'],
            'F1_With_Distance': result['metrics_with']['f1'],
            'F1_Improvement_Pct': result['improvements']['f1_improvement_pct'],
            'ROC_AUC_Without_Distance': result['metrics_without']['roc_auc'],
            'ROC_AUC_With_Distance': result['metrics_with']['roc_auc'],
            'ROC_AUC_Improvement_Pct': result['improvements']['roc_auc_improvement_pct']
        })
    
    comparison_df = pd.DataFrame(comparison_summary)
    comparison_df.to_csv('../../models/distance_feature_comparison.csv', index=False)
    
    print("Models and comparison results saved!")
    return comparison_df

def main():
    """Main function to run the entire pipeline"""
    print("🚀 Starting Credit Card Fraud Detection Model Training")
    print("=" * 60)
    print("🏆 Highlighting Distance Feature Innovation")
    print("=" * 60)
    
    # Load and preprocess data
    X_without, X_with, y = load_and_preprocess_data()
    
    # Train models with comparison
    results = train_models_with_comparison(X_without, X_with, y)
    
    # Create visualizations
    create_innovation_comparison_plot(results)
    create_roc_curves_comparison(results)
    
    # Save models and results
    comparison_df = save_models_and_results(results)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 DISTANCE FEATURE INNOVATION SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    # Find best improvement
    best_improvement = comparison_df.loc[comparison_df['F1_Improvement_Pct'].idxmax()]
    print(f"\n🏆 Best Improvement: {best_improvement['Model']}")
    print(f"   F1-Score Improvement: {best_improvement['F1_Improvement_Pct']:.2f}%")
    print(f"   ROC-AUC Improvement: {best_improvement['ROC_AUC_Improvement_Pct']:.2f}%")
    
    print("\n✅ Training completed successfully!")
    print("📁 Models saved in 'models/' directory")
    print("📊 Visualizations saved in 'visualizations/' directory")
    print("🎯 Distance feature innovation documented!")

if __name__ == "__main__":
    main() 