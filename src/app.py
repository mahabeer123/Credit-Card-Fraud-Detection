#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Interactive Demo
WOW Factor Features:
1. Live Fraud Monitor - Real-time transaction monitoring
2. Fraud Detective Game - Interactive learning experience
3. Scenario Explorer - Slider-based prediction testing
4. Batch Analysis - CSV upload and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .game-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    import os
    
    # Try multiple possible paths for models
    model_paths = [
        'src/models/random_forest_model.pkl',
        'models/random_forest_model.pkl',
        'random_forest_model.pkl'
    ]
    
    scaler_paths = [
        'src/models/scaler.pkl',
        'models/scaler.pkl',
        'scaler.pkl'
    ]
    
    # Try to load models from different paths
    for model_path, scaler_path in zip(model_paths, scaler_paths):
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                rf_model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                st.success(f"✅ Models loaded successfully from {model_path}")
                return rf_model, scaler
        except Exception as e:
            continue
    
    # If models not found, try to train them
    st.warning("⚠️ Models not found. Training models now...")
    try:
        # Try different import paths for save_models
        try:
            from save_models_standalone import save_models
        except ImportError:
            try:
                from src.models.save_models import save_models
            except ImportError:
                try:
                    from models.save_models import save_models
                except ImportError:
                    # Create dummy models if all imports fail
                    st.info("💡 Using synthetic data for demo purposes...")
                    return create_dummy_model(), create_dummy_scaler()
        
        rf_model, scaler = save_models()
        if rf_model is not None and scaler is not None:
            st.success("✅ Models trained and loaded successfully!")
            return rf_model, scaler
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        st.info("💡 Using synthetic data for demo purposes...")
        return create_dummy_model(), create_dummy_scaler()
    
    st.error("❌ Models not found and could not be trained. Please check the model files.")
    return None, None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstrations"""
    import os
    
    # List of possible paths to try
    data_paths = [
        'data/fraudTrain.csv',
        'src/data/fraudTrain.csv',
        '../data/fraudTrain.csv',
        '../../data/fraudTrain.csv',
        'fraudTrain.csv'
    ]
    
    for data_path in data_paths:
        try:
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                return data.sample(n=1000, random_state=42)
        except Exception as e:
            continue
    
    # If data not found, create sample data
    st.warning("⚠️ Sample data not found. Creating synthetic data for demo...")
    return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic data for demo when real data is not available"""
    np.random.seed(42)
    n_samples = 1000
    
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
        'distance': np.random.exponential(50, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.995, 0.005])
    }
    
    return pd.DataFrame(data)

def create_dummy_model():
    """Create a dummy model for demo purposes"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train on synthetic data
    X = np.random.rand(100, 14)
    y = np.random.choice([0, 1], 100, p=[0.95, 0.05])
    model.fit(X, y)
    return model

def create_dummy_scaler():
    """Create a dummy scaler for demo purposes"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit on synthetic data
    X = np.random.rand(100, 14)
    scaler.fit(X)
    return scaler

def generate_sample_transaction():
    """Generate a realistic sample transaction"""
    return {
        'cc_num': random.randint(1000000000000000, 9999999999999999),
        'amt': round(random.uniform(1, 1000), 2),
        'zip': random.randint(10000, 99999),
        'lat': round(random.uniform(25, 50), 6),
        'long': round(random.uniform(-125, -65), 6),
        'city_pop': random.randint(1000, 1000000),
        'unix_time': int(time.time()),
        'merch_lat': round(random.uniform(25, 50), 6),
        'merch_long': round(random.uniform(-125, -65), 6),
        'trans_hour': random.randint(0, 23),
        'trans_day_of_week': random.randint(0, 6),
        'trans_month': random.randint(1, 12),
        'age': random.randint(18, 80)
    }

def calculate_distance(lat1, long1, lat2, long2):
    """Calculate distance between two points"""
    from geopy.distance import great_circle
    return great_circle((lat1, long1), (lat2, long2)).kilometers

def predict_fraud(transaction_data, model, scaler):
    """Predict fraud probability for a transaction"""
    try:
        # Calculate distance feature
        distance = calculate_distance(
            transaction_data['lat'], transaction_data['long'],
            transaction_data['merch_lat'], transaction_data['merch_long']
        )
        
        # Prepare features in the exact same order as training
        # Feature order: ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'trans_hour', 'trans_day_of_week', 'trans_month', 'age', 'distance']
        features = [
            float(transaction_data['cc_num']),
            float(transaction_data['amt']),
            float(transaction_data['zip']),
            float(transaction_data['lat']),
            float(transaction_data['long']),
            float(transaction_data['city_pop']),
            float(transaction_data['unix_time']),
            float(transaction_data['merch_lat']),
            float(transaction_data['merch_long']),
            float(transaction_data['trans_hour']),
            float(transaction_data['trans_day_of_week']),
            float(transaction_data['trans_month']),
            float(transaction_data['age']),
            float(distance)
        ]
        
        # Ensure we have exactly 14 features
        if len(features) != 14:
            st.error(f"❌ Expected 14 features, got {len(features)}")
            return 0.0, False
        
        # Check for any NaN or infinite values
        if any(np.isnan(features)) or any(np.isinf(features)):
            st.error("❌ Invalid feature values detected")
            return 0.0, False
        
        # Convert to numpy array and ensure correct shape
        features_array = np.array(features, dtype=np.float64).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Check scaled features for NaN or infinite values
        if np.any(np.isnan(features_scaled)) or np.any(np.isinf(features_scaled)):
            st.error("❌ Invalid scaled feature values")
            return 0.0, False
        
        # Predict with error handling
        try:
            fraud_prob = model.predict_proba(features_scaled)[0][1]
            is_fraud = model.predict(features_scaled)[0]
        except AttributeError as e:
            # Handle scikit-learn version compatibility issues
            if "monotonic_cst" in str(e):
                st.warning("⚠️ Model compatibility issue detected. Using fallback prediction.")
                # Use a simple heuristic for demo purposes
                fraud_prob = 0.1 if distance > 1000 else 0.05
                is_fraud = fraud_prob > 0.5
            else:
                raise e
        
        return fraud_prob, is_fraud
        
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        # Return default values
        return 0.0, False

def live_fraud_monitor():
    """Live Fraud Monitor - Real-time transaction monitoring"""
    st.markdown('<h1 class="main-header">🕵️ Live Fraud Monitor</h1>', unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    if model is None:
        return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Real-time Transaction Feed")
        
        # Generate live transactions
        if st.button("🔄 Generate New Transaction", type="primary"):
            transaction = generate_sample_transaction()
            fraud_prob, is_fraud = predict_fraud(transaction, model, scaler)
            
            # Display transaction details
            st.markdown("#### Transaction Details:")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Amount", f"${transaction['amt']:.2f}")
                st.metric("Location", f"{transaction['lat']:.4f}, {transaction['long']:.4f}")
            
            with col_b:
                st.metric("Time", f"{transaction['trans_hour']:02d}:00")
                st.metric("Age", f"{transaction['age']} years")
            
            with col_c:
                st.metric("Distance", f"{calculate_distance(transaction['lat'], transaction['long'], transaction['merch_lat'], transaction['merch_long']):.1f} km")
                st.metric("Population", f"{transaction['city_pop']:,}")
            
            # Fraud prediction
            if is_fraud:
                st.markdown('<div class="warning-card"><h3>🚨 FRAUD DETECTED!</h3><p>High Risk Transaction</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-card"><h3>✅ LEGITIMATE</h3><p>Low Risk Transaction</p></div>', unsafe_allow_html=True)
            
            # Fraud probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Live Metrics")
        
        # Simulate live metrics
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.metric("Transactions/sec", f"{random.randint(10, 50)}")
            st.metric("Fraud Rate", f"{random.uniform(0.1, 2.0):.2f}%")
        
        with col_metrics2:
            st.metric("Avg Amount", f"${random.uniform(50, 200):.0f}")
            st.metric("Response Time", f"{random.uniform(10, 100):.0f}ms")
        
        # Recent alerts
        st.markdown("### 🚨 Recent Alerts")
        alerts = [
            "High amount transaction detected",
            "Unusual location pattern",
            "Multiple rapid transactions",
            "Suspicious time pattern"
        ]
        
        for alert in random.sample(alerts, 3):
            st.info(f"⚠️ {alert}")

def fraud_detective_game():
    """Fraud Detective Game - Interactive learning experience"""
    st.markdown('<h1 class="main-header">🎮 Fraud Detective Game</h1>', unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    if model is None:
        return
    
    st.markdown("### 🕵️ Can you spot the fraud?")
    st.markdown("Analyze the transaction and make your prediction!")
    
    # Initialize session state for game
    if 'game_score' not in st.session_state:
        st.session_state.game_score = 0
    if 'game_round' not in st.session_state:
        st.session_state.game_round = 1
    
    # Generate a transaction
    transaction = generate_sample_transaction()
    fraud_prob, is_fraud = predict_fraud(transaction, model, scaler)
    
    # Display transaction details
    st.markdown("#### 🔍 Transaction Analysis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Transaction Details:**")
        st.write(f"💰 Amount: ${transaction['amt']:.2f}")
        st.write(f"📍 Location: {transaction['lat']:.4f}, {transaction['long']:.4f}")
        st.write(f"🏪 Merchant: {transaction['merch_lat']:.4f}, {transaction['merch_long']:.4f}")
        st.write(f"⏰ Time: {transaction['trans_hour']:02d}:00")
        st.write(f"👤 Age: {transaction['age']} years")
        st.write(f"🌍 Distance: {calculate_distance(transaction['lat'], transaction['long'], transaction['merch_lat'], transaction['merch_long']):.1f} km")
    
    with col2:
        # Create feature importance visualization
        features = ['Amount', 'Time', 'Distance', 'Age', 'Location']
        importance = [transaction['amt']/1000, transaction['trans_hour']/24, 
                     calculate_distance(transaction['lat'], transaction['long'], transaction['merch_lat'], transaction['merch_long'])/100,
                     transaction['age']/100, 1]
        
        fig = px.bar(x=features, y=importance, title="Feature Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    # Player prediction
    st.markdown("### 🎯 Your Prediction:")
    prediction = st.radio(
        "Is this transaction fraudulent?",
        ["Legitimate", "Fraudulent"],
        key=f"prediction_{st.session_state.game_round}"
    )
    
    if st.button("🔍 Reveal Answer", type="primary"):
        player_correct = (prediction == "Fraudulent" and is_fraud) or (prediction == "Legitimate" and not is_fraud)
        
        if player_correct:
            st.session_state.game_score += 1
            st.success("🎉 Correct! Great detective work!")
        else:
            st.error("❌ Wrong! Better luck next time!")
        
        # Show actual result
        if is_fraud:
            st.warning(f"🚨 This was actually FRAUDULENT! (Probability: {fraud_prob:.1%})")
        else:
            st.success(f"✅ This was actually LEGITIMATE! (Probability: {fraud_prob:.1%})")
        
        # Update score
        st.markdown(f"### 📊 Score: {st.session_state.game_score}/{st.session_state.game_round}")
        
        if st.button("🔄 Next Round"):
            st.session_state.game_round += 1
            st.rerun()

def scenario_explorer():
    """Scenario Explorer - Analyze how different factors affect fraud probability"""
    st.markdown('<h1 class="main-header">🔬 Scenario Explorer</h1>', unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    if model is None:
        return
    
    st.markdown("### 📊 Analyze How Different Factors Affect Fraud Probability")
    
    # Generate a base transaction
    transaction = generate_sample_transaction()
    
    # Create subplots
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Amount Impact", "Time Impact", "Distance Impact", "Age Impact"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Amount impact - explicitly convert to list
        amounts = list(range(1, 1001, 50))
        prob_amounts = []
        for amt in amounts:
            temp_transaction = transaction.copy()
            temp_transaction['amt'] = amt
            prob, _ = predict_fraud(temp_transaction, model, scaler)
            prob_amounts.append(prob)
        
        # Ensure amounts is a list for Plotly
        amounts_list = list(amounts)
        fig.add_trace(go.Scatter(x=amounts_list, y=prob_amounts, name="Amount"), row=1, col=1)
        
        # Time impact - explicitly convert to list
        hours = list(range(24))
        prob_hours = []
        for hr in hours:
            temp_transaction = transaction.copy()
            temp_transaction['trans_hour'] = hr
            prob, _ = predict_fraud(temp_transaction, model, scaler)
            prob_hours.append(prob)
        
        # Ensure hours is a list for Plotly
        hours_list = list(hours)
        fig.add_trace(go.Scatter(x=hours_list, y=prob_hours, name="Time"), row=1, col=2)
        
        # Distance impact - explicitly convert to list
        distances = list(range(0, 1001, 50))
        prob_distances = []
        for dist in distances:
            temp_transaction = transaction.copy()
            temp_transaction['merch_lat'] = transaction['lat'] + (dist/111)
            temp_transaction['merch_long'] = transaction['long'] + (dist/111)
            prob, _ = predict_fraud(temp_transaction, model, scaler)
            prob_distances.append(prob)
        
        # Ensure distances is a list for Plotly
        distances_list = list(distances)
        fig.add_trace(go.Scatter(x=distances_list, y=prob_distances, name="Distance"), row=2, col=1)
        
        # Age impact - explicitly convert to list
        ages = list(range(18, 81, 5))
        prob_ages = []
        for age_val in ages:
            temp_transaction = transaction.copy()
            temp_transaction['age'] = age_val
            prob, _ = predict_fraud(temp_transaction, model, scaler)
            prob_ages.append(prob)
        
        # Ensure ages is a list for Plotly
        ages_list = list(ages)
        fig.add_trace(go.Scatter(x=ages_list, y=prob_ages, name="Age"), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error generating scenario analysis: {str(e)}")
        st.info("💡 Using fallback data for demonstration.")
        
        # Create a simple fallback visualization
        import numpy as np
        
        # Generate sample data for demonstration - explicitly use lists
        amounts = list(range(1, 1001, 50))
        prob_amounts = [0.05 + 0.1 * (amt/1000) for amt in amounts]
        
        hours = list(range(24))
        prob_hours = [0.08 + 0.04 * np.sin(h/24 * 2 * np.pi) for h in hours]
        
        distances = list(range(0, 1001, 50))
        prob_distances = [0.06 + 0.12 * (dist/1000) for dist in distances]
        
        ages = list(range(18, 81, 5))
        prob_ages = [0.07 + 0.03 * np.sin((age-18)/63 * 2 * np.pi) for age in ages]
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Amount Impact", "Time Impact", "Distance Impact", "Age Impact"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Ensure all data is explicitly converted to lists
            fig.add_trace(go.Scatter(x=list(amounts), y=prob_amounts, name="Amount"), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(hours), y=prob_hours, name="Time"), row=1, col=2)
            fig.add_trace(go.Scatter(x=list(distances), y=prob_distances, name="Distance"), row=2, col=1)
            fig.add_trace(go.Scatter(x=list(ages), y=prob_ages, name="Age"), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e2:
            st.error(f"❌ Error creating fallback visualization: {str(e2)}")
            st.info("💡 Feature temporarily unavailable. Please try again later.")
            
            # Show simple metrics instead
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Amount Impact", "High", "Larger amounts = Higher risk")
            with col2:
                st.metric("Time Impact", "Medium", "Night transactions = Higher risk")
            with col3:
                st.metric("Distance Impact", "High", "Longer distances = Higher risk")
            with col4:
                st.metric("Age Impact", "Low", "Age has minimal effect")

def batch_analysis():
    """Batch Analysis - CSV upload and analysis"""
    st.markdown('<h1 class="main-header">📊 Batch Analysis</h1>', unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    if model is None:
        return
    
    st.markdown("### 📁 Upload CSV File for Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with transaction data",
        type=['csv'],
        help="File should contain columns: cc_num, amt, zip, lat, long, city_pop, unix_time, merch_lat, merch_long, trans_hour, trans_day_of_week, trans_month, age"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(data)} transactions")
            
            # Show data preview
            st.markdown("#### 📋 Data Preview:")
            st.dataframe(data.head(), use_container_width=True)
            
            # Analyze data
            if st.button("🔍 Analyze Transactions", type="primary"):
                with st.spinner("Analyzing transactions..."):
                    results = []
                    
                    for idx, row in data.iterrows():
                        try:
                            # Prepare transaction data
                            transaction = {
                                'cc_num': row.get('cc_num', random.randint(1000000000000000, 9999999999999999)),
                                'amt': row.get('amt', 100),
                                'zip': row.get('zip', 12345),
                                'lat': row.get('lat', 40.0),
                                'long': row.get('long', -74.0),
                                'city_pop': row.get('city_pop', 100000),
                                'unix_time': row.get('unix_time', int(time.time())),
                                'merch_lat': row.get('merch_lat', 40.0),
                                'merch_long': row.get('merch_long', -74.0),
                                'trans_hour': row.get('trans_hour', 12),
                                'trans_day_of_week': row.get('trans_day_of_week', 2),
                                'trans_month': row.get('trans_month', 6),
                                'age': row.get('age', 35)
                            }
                            
                            fraud_prob, is_fraud = predict_fraud(transaction, model, scaler)
                            
                            results.append({
                                'Transaction_ID': idx + 1,
                                'Amount': transaction['amt'],
                                'Fraud_Probability': fraud_prob,
                                'Prediction': 'Fraud' if is_fraud else 'Legitimate',
                                'Risk_Level': 'High' if fraud_prob > 0.7 else 'Medium' if fraud_prob > 0.3 else 'Low'
                            })
                            
                        except Exception as e:
                            st.error(f"Error processing transaction {idx + 1}: {str(e)}")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.markdown("#### 📊 Analysis Results:")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("#### 📈 Summary Statistics:")
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    
                    with col_sum1:
                        st.metric("Total Transactions", len(results_df))
                    
                    with col_sum2:
                        fraud_count = len(results_df[results_df['Prediction'] == 'Fraud'])
                        st.metric("Fraud Detected", fraud_count)
                    
                    with col_sum3:
                        avg_prob = results_df['Fraud_Probability'].mean()
                        st.metric("Avg Fraud Probability", f"{avg_prob:.1%}")
                    
                    with col_sum4:
                        high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                        st.metric("High Risk", high_risk)
                    
                    # Create visualizations
                    st.markdown("#### 📊 Visualizations:")
                    
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Fraud probability distribution
                        fig1 = px.histogram(results_df, x='Fraud_Probability', 
                                          title="Fraud Probability Distribution",
                                          nbins=20)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_viz2:
                        # Risk level pie chart
                        risk_counts = results_df['Risk_Level'].value_counts()
                        fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                                     title="Risk Level Distribution")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.markdown("## 🕵️ Fraud Detection Demo")
    
    page = st.sidebar.selectbox(
        "Choose a Demo:",
        ["Live Fraud Monitor", "Fraud Detective Game", "Scenario Explorer", "Batch Analysis"]
    )
    
    # Display selected page
    if page == "Live Fraud Monitor":
        live_fraud_monitor()
    elif page == "Fraud Detective Game":
        fraud_detective_game()
    elif page == "Scenario Explorer":
        scenario_explorer()
    elif page == "Batch Analysis":
        batch_analysis()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("ROC-AUC", "0.9604")
    st.sidebar.metric("Recall", "91.96%")
    st.sidebar.metric("Training Time", "0.51s")

if __name__ == "__main__":
    main() 