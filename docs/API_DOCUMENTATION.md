# 🔌 API Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)
8. [SDK Examples](#sdk-examples)

---

## 🎯 Overview

The Credit Card Fraud Detection API provides real-time fraud detection capabilities through a RESTful interface. The API is designed for high-performance, low-latency fraud detection with comprehensive model explainability.

### **Base URL**
```
https://credit-card-fraud-detection-framework.streamlit.app/api/v1
```

### **Content Type**
```
Content-Type: application/json
```

---

## 🔐 Authentication

### **API Key Authentication**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.example.com/v1/predict
```

### **Authentication Headers**
| Header | Type | Description |
|--------|------|-------------|
| `Authorization` | string | Bearer token for API access |
| `Content-Type` | string | Must be `application/json` |
| `X-Request-ID` | string | Unique request identifier (optional) |

---

## 🚀 Endpoints

### **1. Fraud Detection Prediction**

#### **POST** `/predict`

Predicts fraud probability for a single transaction.

**Request Body:**
```json
{
  "transaction": {
    "cc_num": 1234567890123456,
    "amt": 150.50,
    "zip": 12345,
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 8336817,
    "unix_time": 1640995200,
    "merch_lat": 40.7589,
    "merch_long": -73.9851,
    "trans_hour": 14,
    "trans_day_of_week": 2,
    "trans_month": 1,
    "age": 35
  }
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "fraud_probability": 0.234,
    "is_fraud": false,
    "risk_level": "LOW",
    "confidence": 0.89
  },
  "features": {
    "amount_risk": 0.15,
    "time_risk": 0.05,
    "distance_risk": 0.08,
    "age_risk": 0.02
  },
  "explanation": {
    "top_factors": [
      "Transaction amount is moderate",
      "Transaction time is during business hours",
      "Distance is within normal range"
    ],
    "recommendation": "APPROVE"
  },
  "metadata": {
    "model_version": "1.0.0",
    "prediction_time": "2024-01-01T12:00:00Z",
    "request_id": "req_123456789"
  }
}
```

### **2. Batch Prediction**

#### **POST** `/predict/batch`

Predicts fraud probability for multiple transactions.

**Request Body:**
```json
{
  "transactions": [
    {
      "cc_num": 1234567890123456,
      "amt": 150.50,
      "zip": 12345,
      "lat": 40.7128,
      "long": -74.0060,
      "city_pop": 8336817,
      "unix_time": 1640995200,
      "merch_lat": 40.7589,
      "merch_long": -73.9851,
      "trans_hour": 14,
      "trans_day_of_week": 2,
      "trans_month": 1,
      "age": 35
    },
    {
      "cc_num": 9876543210987654,
      "amt": 2500.00,
      "zip": 54321,
      "lat": 34.0522,
      "long": -118.2437,
      "city_pop": 3979576,
      "unix_time": 1640995200,
      "merch_lat": 34.0522,
      "merch_long": -118.2437,
      "trans_hour": 23,
      "trans_day_of_week": 6,
      "trans_month": 1,
      "age": 25
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "transaction_id": 0,
      "fraud_probability": 0.234,
      "is_fraud": false,
      "risk_level": "LOW",
      "confidence": 0.89
    },
    {
      "transaction_id": 1,
      "fraud_probability": 0.876,
      "is_fraud": true,
      "risk_level": "HIGH",
      "confidence": 0.92
    }
  ],
  "summary": {
    "total_transactions": 2,
    "fraud_count": 1,
    "fraud_rate": 0.5,
    "average_risk": 0.555
  },
  "metadata": {
    "model_version": "1.0.0",
    "processing_time": "0.045s",
    "request_id": "req_123456790"
  }
}
```

### **3. Model Information**

#### **GET** `/model/info`

Returns information about the deployed model.

**Response:**
```json
{
  "success": true,
  "model": {
    "name": "Random Forest Classifier",
    "version": "1.0.0",
    "training_date": "2024-01-01T00:00:00Z",
    "performance": {
      "roc_auc": 0.9604,
      "recall": 0.9196,
      "precision": 0.0132,
      "f1_score": 0.0260
    },
    "features": [
      "amount",
      "trans_hour",
      "unix_time",
      "trans_month",
      "city_pop",
      "distance",
      "age",
      "lat",
      "long",
      "merch_lat",
      "merch_long",
      "zip",
      "trans_day_of_week",
      "cc_num"
    ],
    "feature_importance": {
      "amount": 0.3775,
      "trans_hour": 0.3180,
      "unix_time": 0.0484,
      "trans_month": 0.0300,
      "city_pop": 0.0292
    }
  },
  "metadata": {
    "request_id": "req_123456791"
  }
}
```

### **4. Health Check**

#### **GET** `/health`

Returns the health status of the API.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": "24h 30m 15s",
  "services": {
    "model": "operational",
    "database": "operational",
    "cache": "operational"
  }
}
```

---

## 📊 Data Models

### **Transaction Object**
```json
{
  "cc_num": "integer (16 digits)",
  "amt": "float (transaction amount)",
  "zip": "integer (5 digits)",
  "lat": "float (customer latitude)",
  "long": "float (customer longitude)",
  "city_pop": "integer (city population)",
  "unix_time": "integer (unix timestamp)",
  "merch_lat": "float (merchant latitude)",
  "merch_long": "float (merchant longitude)",
  "trans_hour": "integer (0-23)",
  "trans_day_of_week": "integer (0-6)",
  "trans_month": "integer (1-12)",
  "age": "integer (customer age)"
}
```

### **Prediction Response**
```json
{
  "fraud_probability": "float (0-1)",
  "is_fraud": "boolean",
  "risk_level": "string (LOW|MEDIUM|HIGH)",
  "confidence": "float (0-1)"
}
```

### **Risk Levels**
- **LOW**: 0.0 - 0.3 (Green)
- **MEDIUM**: 0.3 - 0.7 (Yellow)
- **HIGH**: 0.7 - 1.0 (Red)

---

## ❌ Error Handling

### **Error Response Format**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid transaction data",
    "details": {
      "field": "amount",
      "issue": "Amount must be positive"
    }
  },
  "metadata": {
    "request_id": "req_123456792",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### **Error Codes**
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `AUTHENTICATION_ERROR` | 401 | Invalid API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `MODEL_ERROR` | 500 | Model prediction failed |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### **Common Error Scenarios**
```json
// Missing required field
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Missing required field: amount"
  }
}

// Invalid data type
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid data type for amount: expected float, got string"
  }
}

// Out of range value
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Amount must be between 0.01 and 10000.00"
  }
}
```

---

## ⏱️ Rate Limiting

### **Rate Limits**
- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1000 requests/hour
- **Enterprise**: Custom limits

### **Rate Limit Headers**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640998800
```

### **Rate Limit Exceeded Response**
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 3600 seconds."
  }
}
```

---

## 💡 Examples

### **Python SDK Example**
```python
import requests
import json

# API Configuration
API_BASE_URL = "https://api.example.com/v1"
API_KEY = "your_api_key_here"

# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Single prediction
def predict_fraud(transaction_data):
    url = f"{API_BASE_URL}/predict"
    payload = {"transaction": transaction_data}
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.text}")

# Example usage
transaction = {
    "cc_num": 1234567890123456,
    "amt": 150.50,
    "zip": 12345,
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 8336817,
    "unix_time": 1640995200,
    "merch_lat": 40.7589,
    "merch_long": -73.9851,
    "trans_hour": 14,
    "trans_day_of_week": 2,
    "trans_month": 1,
    "age": 35
}

result = predict_fraud(transaction)
print(f"Fraud Probability: {result['prediction']['fraud_probability']}")
print(f"Risk Level: {result['prediction']['risk_level']}")
```

### **JavaScript Example**
```javascript
// API Configuration
const API_BASE_URL = "https://api.example.com/v1";
const API_KEY = "your_api_key_here";

// Headers
const headers = {
    "Authorization": `Bearer ${API_KEY}`,
    "Content-Type": "application/json"
};

// Single prediction
async function predictFraud(transactionData) {
    const url = `${API_BASE_URL}/predict`;
    const payload = { transaction: transactionData };
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            return await response.json();
        } else {
            throw new Error(`API Error: ${response.statusText}`);
        }
    } catch (error) {
        console.error("Prediction failed:", error);
        throw error;
    }
}

// Example usage
const transaction = {
    cc_num: 1234567890123456,
    amt: 150.50,
    zip: 12345,
    lat: 40.7128,
    long: -74.0060,
    city_pop: 8336817,
    unix_time: 1640995200,
    merch_lat: 40.7589,
    merch_long: -73.9851,
    trans_hour: 14,
    trans_day_of_week: 2,
    trans_month: 1,
    age: 35
};

predictFraud(transaction)
    .then(result => {
        console.log(`Fraud Probability: ${result.prediction.fraud_probability}`);
        console.log(`Risk Level: ${result.prediction.risk_level}`);
    })
    .catch(error => {
        console.error("Error:", error);
    });
```

### **cURL Examples**
```bash
# Single prediction
curl -X POST "https://api.example.com/v1/predict" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction": {
         "cc_num": 1234567890123456,
         "amt": 150.50,
         "zip": 12345,
         "lat": 40.7128,
         "long": -74.0060,
         "city_pop": 8336817,
         "unix_time": 1640995200,
         "merch_lat": 40.7589,
         "merch_long": -73.9851,
         "trans_hour": 14,
         "trans_day_of_week": 2,
         "trans_month": 1,
         "age": 35
       }
     }'

# Model information
curl -X GET "https://api.example.com/v1/model/info" \
     -H "Authorization: Bearer YOUR_API_KEY"

# Health check
curl -X GET "https://api.example.com/v1/health"
```

---

## 🔧 SDK Examples

### **Python SDK Installation**
```bash
pip install fraud-detection-sdk
```

### **Python SDK Usage**
```python
from fraud_detection import FraudDetectionClient

# Initialize client
client = FraudDetectionClient(api_key="your_api_key")

# Single prediction
transaction = {
    "cc_num": 1234567890123456,
    "amt": 150.50,
    # ... other fields
}

result = client.predict(transaction)
print(f"Fraud Probability: {result.fraud_probability}")

# Batch prediction
transactions = [transaction1, transaction2, transaction3]
results = client.predict_batch(transactions)

# Model information
model_info = client.get_model_info()
print(f"Model Version: {model_info.version}")
```

### **JavaScript SDK Installation**
```bash
npm install fraud-detection-sdk
```

### **JavaScript SDK Usage**
```javascript
import { FraudDetectionClient } from 'fraud-detection-sdk';

// Initialize client
const client = new FraudDetectionClient('your_api_key');

// Single prediction
const transaction = {
    cc_num: 1234567890123456,
    amt: 150.50,
    // ... other fields
};

const result = await client.predict(transaction);
console.log(`Fraud Probability: ${result.fraud_probability}`);

// Batch prediction
const transactions = [transaction1, transaction2, transaction3];
const results = await client.predictBatch(transactions);

// Model information
const modelInfo = await client.getModelInfo();
console.log(`Model Version: ${modelInfo.version}`);
```

---

*This API documentation provides comprehensive coverage of all endpoints and usage patterns for the Credit Card Fraud Detection system.* 