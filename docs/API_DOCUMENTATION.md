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
      "age": 28
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
      "risk_level": "LOW"
    },
    {
      "transaction_id": 1,
      "fraud_probability": 0.876,
      "is_fraud": true,
      "risk_level": "HIGH"
    }
  ],
  "summary": {
    "total_transactions": 2,
    "fraud_count": 1,
    "fraud_rate": 0.5,
    "average_risk_score": 0.555
  }
}
```

### **3. Model Information**

#### **GET** `/model/info`

Returns information about the current model.

**Response:**
```json
{
  "success": true,
  "model_info": {
    "model_type": "Random Forest",
    "version": "1.0.0",
    "training_date": "2024-01-15",
    "performance": {
      "roc_auc": 0.9604,
      "recall": 0.9196,
      "precision": 0.0132,
      "f1_score": 0.0260
    },
    "features": [
      "amt",
      "trans_hour",
      "unix_time",
      "trans_month",
      "city_pop",
      "distance",
      "age",
      "trans_day_of_week"
    ]
  }
}
```

### **4. Feature Importance**

#### **GET** `/model/features`

Returns feature importance information.

**Response:**
```json
{
  "success": true,
  "feature_importance": [
    {
      "feature": "amt",
      "importance": 0.3775,
      "description": "Transaction amount"
    },
    {
      "feature": "trans_hour",
      "importance": 0.3180,
      "description": "Transaction hour"
    },
    {
      "feature": "unix_time",
      "importance": 0.0484,
      "description": "Unix timestamp"
    }
  ]
}
```

---

## 📊 Data Models

### **Transaction Model**
```json
{
  "cc_num": "integer",
  "amt": "float",
  "zip": "integer",
  "lat": "float",
  "long": "float",
  "city_pop": "integer",
  "unix_time": "integer",
  "merch_lat": "float",
  "merch_long": "float",
  "trans_hour": "integer",
  "trans_day_of_week": "integer",
  "trans_month": "integer",
  "age": "integer"
}
```

### **Prediction Response Model**
```json
{
  "success": "boolean",
  "prediction": {
    "fraud_probability": "float",
    "is_fraud": "boolean",
    "risk_level": "string",
    "confidence": "float"
  },
  "features": {
    "amount_risk": "float",
    "time_risk": "float",
    "distance_risk": "float",
    "age_risk": "float"
  },
  "explanation": {
    "top_factors": ["string"],
    "recommendation": "string"
  }
}
```

---

## ⚠️ Error Handling

### **Error Response Format**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details"
  }
}
```

### **Common Error Codes**
| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_REQUEST` | Invalid request format | 400 |
| `MISSING_FIELDS` | Required fields missing | 400 |
| `INVALID_DATA` | Invalid data values | 400 |
| `MODEL_ERROR` | Model prediction error | 500 |
| `RATE_LIMIT` | Rate limit exceeded | 429 |
| `UNAUTHORIZED` | Authentication required | 401 |

---

## 🚦 Rate Limiting

### **Rate Limits**
- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1000 requests/hour
- **Enterprise**: Custom limits

### **Rate Limit Headers**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## 📝 Examples

### **Python Example**
```python
import requests
import json

# API configuration
API_BASE_URL = "https://credit-card-fraud-detection-framework.streamlit.app/api/v1"
API_KEY = "your_api_key_here"

# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Transaction data
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

# Make prediction request
response = requests.post(
    f"{API_BASE_URL}/predict",
    headers=headers,
    json={"transaction": transaction}
)

# Handle response
if response.status_code == 200:
    result = response.json()
    print(f"Fraud Probability: {result['prediction']['fraud_probability']}")
    print(f"Risk Level: {result['prediction']['risk_level']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### **JavaScript Example**
```javascript
// API configuration
const API_BASE_URL = "https://credit-card-fraud-detection-framework.streamlit.app/api/v1";
const API_KEY = "your_api_key_here";

// Transaction data
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

// Make prediction request
fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ transaction })
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log(`Fraud Probability: ${data.prediction.fraud_probability}`);
        console.log(`Risk Level: ${data.prediction.risk_level}`);
    } else {
        console.error(`Error: ${data.error.message}`);
    }
})
.catch(error => {
    console.error('Request failed:', error);
});
```

### **cURL Example**
```bash
# Single prediction
curl -X POST \
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
  }' \
  https://credit-card-fraud-detection-framework.streamlit.app/api/v1/predict
```

---

## 🔧 SDK Examples

### **Python SDK**
```python
from fraud_detection_sdk import FraudDetectionClient

# Initialize client
client = FraudDetectionClient(api_key="your_api_key_here")

# Single prediction
result = client.predict(transaction)
print(f"Fraud Probability: {result.fraud_probability}")

# Batch prediction
results = client.predict_batch(transactions)
for result in results:
    print(f"Transaction {result.transaction_id}: {result.risk_level}")
```

### **JavaScript SDK**
```javascript
import { FraudDetectionClient } from 'fraud-detection-sdk';

// Initialize client
const client = new FraudDetectionClient('your_api_key_here');

// Single prediction
const result = await client.predict(transaction);
console.log(`Fraud Probability: ${result.fraud_probability}`);

// Batch prediction
const results = await client.predictBatch(transactions);
results.forEach(result => {
    console.log(`Transaction ${result.transactionId}: ${result.riskLevel}`);
});
```

---

## 📞 Support

For API support and questions:
- **Documentation**: [API Documentation](docs/API_DOCUMENTATION.md)
- **GitHub Issues**: [Report bugs](https://github.com/mahabeer123/Credit-Card-Fraud-Detection/issues)
- **Email**: support@fraud-detection.com

---

## 📄 License

This API is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 
*This API documentation provides comprehensive coverage of all endpoints and usage patterns for the Credit Card Fraud Detection system.* 