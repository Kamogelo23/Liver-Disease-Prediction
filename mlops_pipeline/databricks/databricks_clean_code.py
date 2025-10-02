#!/usr/bin/env python3
"""
Clean Databricks Code - No Unicode Characters
Copy and paste this code directly into Databricks notebooks
"""

# =============================================================================
# CELL 1: Setup & Data Generation
# =============================================================================

# Install packages
%pip install plotly mlflow scikit-learn

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Generate realistic liver disease data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Age': np.random.normal(50, 15, n_samples).astype(int),
    'Sex': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
    'ALB': np.random.normal(4.2, 0.6, n_samples).round(2),
    'ALP': np.random.normal(95, 35, n_samples).round(0),
    'ALT': np.random.normal(35, 25, n_samples).round(0),
    'AST': np.random.normal(30, 20, n_samples).round(0),
    'BIL': np.random.normal(0.8, 0.4, n_samples).round(2),
    'CHE': np.random.normal(8.5, 2.0, n_samples).round(2),
    'CHOL': np.random.normal(200, 50, n_samples).round(0),
    'CREA': np.random.normal(1.0, 0.3, n_samples).round(2),
    'GGT': np.random.normal(40, 30, n_samples).round(0),
    'PROT': np.random.normal(7.0, 0.8, n_samples).round(2),
})

# Add realistic disease categories
def assign_category(row):
    if row['Age'] < 40 and row['ALT'] < 30 and row['AST'] < 25:
        return 'Blood donor'
    elif row['ALT'] > 60 or row['AST'] > 50:
        return 'Hepatitis'
    elif row['ALB'] < 3.5 and row['BIL'] > 1.5:
        return 'Cirrhosis'
    else:
        return np.random.choice(['Blood donor', 'Cirrhosis', 'Fibrosis', 'Hepatitis'], 
                               p=[0.4, 0.2, 0.2, 0.2])

data['Category'] = data.apply(assign_category, axis=1)

print(f"Dataset created: {data.shape}")
print(f"Target distribution:")
print(data['Category'].value_counts())

# =============================================================================
# CELL 2: Model Training with MLflow
# =============================================================================

# Setup MLflow - Fixed for Databricks
mlflow.set_experiment("/liver_disease_prediction")

# Prepare data
le_sex = LabelEncoder()
le_target = LabelEncoder()

data['Sex_encoded'] = le_sex.fit_transform(data['Sex'])
data['Category_encoded'] = le_target.fit_transform(data['Category'])

feature_cols = ['Age', 'Sex_encoded', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
X = data[feature_cols]
y = data['Category_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model with MLflow tracking
with mlflow.start_run(run_name="random_forest_liver_disease"):
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model", registered_model_name="liver_disease_classifier")
    
    print(f"Model trained!")
    print(f"   Train Accuracy: {train_accuracy:.3f}")
    print(f"   Test Accuracy: {test_accuracy:.3f}")
    print(f"   Model logged to MLflow!")

# =============================================================================
# CELL 3: Prediction Function & API
# =============================================================================

# Create prediction function
def predict_liver_disease(features_dict):
    """Predict liver disease category from features."""
    # Prepare features
    features_df = pd.DataFrame([features_dict])
    features_df['Sex_encoded'] = le_sex.transform(features_df['Sex'])
    X = features_df[feature_cols]
    
    # Make prediction
    prediction_encoded = model.predict(X)[0]
    prediction = le_target.inverse_transform([prediction_encoded])[0]
    
    # Get probabilities
    proba_encoded = model.predict_proba(X)[0]
    probabilities = {
        le_target.classes_[i]: float(proba_encoded[i]) 
        for i in range(len(le_target.classes_))
    }
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'confidence': float(max(proba_encoded))
    }

# Test prediction
sample_patient = {
    'Age': 45, 'Sex': 'M', 'ALB': 4.0, 'ALP': 100, 'ALT': 40,
    'AST': 35, 'BIL': 1.0, 'CHE': 8.0, 'CHOL': 200, 'CREA': 1.1,
    'GGT': 45, 'PROT': 7.2
}

result = predict_liver_disease(sample_patient)
print("Sample Prediction:")
print(f"   Patient: 45-year-old male")
print(f"   Prediction: {result['prediction']}")
print(f"   Confidence: {result['confidence']:.3f}")
print(f"   Probabilities: {result['probabilities']}")

# =============================================================================
# CELL 4: Model Registry & Deployment
# =============================================================================

# Save model for deployment
model_package = {
    'model': model,
    'label_encoders': {'sex': le_sex, 'target': le_target},
    'feature_columns': feature_cols,
    'metadata': {
        'accuracy': test_accuracy,
        'trained_at': pd.Timestamp.now().isoformat()
    }
}

joblib.dump(model_package, '/tmp/liver_disease_model.pkl')

print("DEPLOYMENT READY!")
print("=" * 30)
print(f"Model Accuracy: {test_accuracy:.3f}")
print(f"MLflow Experiment: Created")
print(f"Model Registry: Ready")
print(f"API Function: Working")
print(f"Model Saved: /tmp/liver_disease_model.pkl")

print("\nNext Steps for Production:")
print("1. Go to MLflow UI - Register Model")
print("2. Create Model Serving Endpoint")
print("3. Test API with real data")
print("4. Set up monitoring")

print("\nPerfect for Nedbank interview demo!")
