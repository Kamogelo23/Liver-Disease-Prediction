#!/usr/bin/env python3
"""
Databricks Notebook Template for Liver Disease Prediction
Copy and paste this code into Databricks Community Edition notebooks
"""

# =============================================================================
# NOTEBOOK 1: DATA EXPLORATION & ANALYSIS
# =============================================================================

# Cell 1: Setup and Imports
%pip install plotly>=5.17.0
%pip install mlflow>=2.8.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("📊 Data Exploration Environment Ready!")

# Cell 2: Generate Sample Data
np.random.seed(42)
n_samples = 1000

# Generate realistic liver disease data
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

# Add disease categories with realistic patterns
def assign_category(row):
    if row['Age'] < 40 and row['ALT'] < 30 and row['AST'] < 25:
        return 'Blood donor'
    elif row['ALT'] > 60 or row['AST'] > 50:
        return 'Hepatitis'
    elif row['ALB'] < 3.5 and row['BIL'] > 1.5:
        return 'Cirrhosis'
    elif row['ALB'] < 4.0 and (row['ALT'] > 40 or row['AST'] > 35):
        return 'Fibrosis'
    else:
        return np.random.choice(['Blood donor', 'Cirrhosis', 'Fibrosis', 'Hepatitis'], 
                               p=[0.4, 0.2, 0.2, 0.2])

data['Category'] = data.apply(assign_category, axis=1)

print(f"✅ Dataset created: {data.shape}")
print(f"📊 Target distribution:")
print(data['Category'].value_counts())

# Cell 3: Interactive Visualizations
# Target distribution
fig = px.pie(data, values='Category', names='Category', 
              title='Distribution of Liver Disease Categories',
              color_discrete_sequence=px.colors.qualitative.Set3)
fig.show()

# Feature distributions
key_features = ['Age', 'ALB', 'ALT', 'AST', 'BIL', 'GGT']
fig = make_subplots(rows=2, cols=3, subplot_titles=key_features)

for i, feature in enumerate(key_features):
    row = i // 3 + 1
    col = i % 3 + 1
    
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category][feature].dropna()
        fig.add_trace(
            go.Box(y=category_data, name=category, boxpoints='outliers'),
            row=row, col=col
        )

fig.update_layout(title_text="Feature Distributions by Disease Category", height=600)
fig.show()

# Save data for next notebook
data.to_csv('/tmp/liver_disease_data.csv', index=False)
print("💾 Data saved for model training")

# =============================================================================
# NOTEBOOK 2: MODEL TRAINING WITH MLFLOW
# =============================================================================

# Cell 1: Setup
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

# Load data
data = pd.read_csv('/tmp/liver_disease_data.csv')
print(f"✅ Data loaded: {data.shape}")

# Cell 2: Data Preprocessing
# Handle missing values
data_clean = data.copy()
numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if data_clean[col].isnull().sum() > 0:
        data_clean[col].fillna(data_clean[col].median(), inplace=True)

# Encode categorical variables
le_sex = LabelEncoder()
data_clean['Sex_encoded'] = le_sex.fit_transform(data_clean['Sex'])

le_target = LabelEncoder()
data_clean['Category_encoded'] = le_target.fit_transform(data_clean['Category'])

# Prepare features
feature_cols = ['Age', 'Sex_encoded', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
X = data_clean[feature_cols]
y = data_clean['Category_encoded']

print(f"✅ Features prepared: {X.shape}")

# Cell 3: MLflow Setup
experiment_name = f"liver_disease_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass
mlflow.set_experiment(experiment_name)

print(f"🎯 MLflow experiment: {experiment_name}")

# Cell 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# Cell 5: Model Training
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    print(f"🚀 Training {name}...")
    
    with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%H%M%S')}"):
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Log to MLflow
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        
        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name=f"liver_disease_{name}")
        
        results[name] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"   ✅ {name} - Test: {test_accuracy:.3f}, CV: {cv_scores.mean():.3f}")

# Cell 6: Model Comparison
comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Test Accuracy': result['test_accuracy'],
        'CV Score': result['cv_mean'],
        'CV Std': result['cv_std']
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('Test Accuracy', ascending=False)
print("📊 Model Performance:")
display(comparison_df)

best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"🏆 Best Model: {best_model_name}")

# Cell 7: Save Best Model
model_package = {
    'model': best_model,
    'label_encoders': {'sex': le_sex, 'target': le_target},
    'feature_columns': feature_cols,
    'metadata': {
        'model_name': best_model_name,
        'test_accuracy': comparison_df.iloc[0]['Test Accuracy'],
        'trained_at': datetime.now().isoformat(),
        'target_classes': le_target.classes_.tolist()
    }
}

model_path = f"/tmp/liver_disease_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model_package, model_path)
print(f"✅ Model saved: {model_path}")

# =============================================================================
# NOTEBOOK 3: MODEL SERVING & API
# =============================================================================

# Cell 1: Load Model
import joblib
from datetime import datetime

# Load the trained model
model_files = [f for f in os.listdir('/tmp') if f.startswith('liver_disease_model_')]
latest_model = max(model_files, key=lambda x: x.split('_')[-1].split('.')[0])
model_path = f"/tmp/{latest_model}"

model_package = joblib.load(model_path)
print(f"✅ Model loaded: {model_package['metadata']['model_name']}")

# Cell 2: Prediction Function
def predict_liver_disease(features_dict):
    """Predict liver disease category from features."""
    model = model_package['model']
    le_sex = model_package['label_encoders']['sex']
    le_target = model_package['label_encoders']['target']
    feature_cols = model_package['feature_columns']
    
    # Prepare features
    features_df = pd.DataFrame([features_dict])
    features_df['Sex_encoded'] = le_sex.transform(features_df['Sex'])
    X = features_df[feature_cols]
    
    # Make prediction
    prediction_encoded = model.predict(X)[0]
    prediction = le_target.inverse_transform([prediction_encoded])[0]
    
    # Get probabilities
    probabilities = None
    if hasattr(model, 'predict_proba'):
        proba_encoded = model.predict_proba(X)[0]
        probabilities = {
            le_target.classes_[i]: float(proba_encoded[i]) 
            for i in range(len(le_target.classes_))
        }
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'model_used': model_package['metadata']['model_name']
    }

print("✅ Prediction function ready!")

# Cell 3: Test Predictions
# Sample patient
patient_data = {
    'Age': 45, 'Sex': 'M', 'ALB': 4.0, 'ALP': 100, 'ALT': 40,
    'AST': 35, 'BIL': 1.0, 'CHE': 8.0, 'CHOL': 200, 'CREA': 1.1,
    'GGT': 45, 'PROT': 7.2
}

result = predict_liver_disease(patient_data)
print("🎯 Sample Prediction:")
print(f"   Patient: {patient_data}")
print(f"   Prediction: {result['prediction']}")
print(f"   Probabilities: {result['probabilities']}")

# Cell 4: API Endpoint Simulation
class ModelServer:
    def __init__(self, model_path):
        self.model_package = joblib.load(model_path)
        self.model = self.model_package['model']
        self.le_sex = self.model_package['label_encoders']['sex']
        self.le_target = self.model_package['label_encoders']['target']
        self.feature_cols = self.model_package['feature_columns']
        self.request_count = 0
    
    def predict(self, features):
        self.request_count += 1
        return predict_liver_disease(features)
    
    def health_check(self):
        return {
            'status': 'healthy',
            'model': self.model_package['metadata']['model_name'],
            'requests': self.request_count
        }

# Initialize server
server = ModelServer(model_path)
print("🚀 Model server ready!")

# Cell 5: API Testing
# Test multiple patients
test_patients = [
    {'Age': 35, 'Sex': 'M', 'ALB': 4.5, 'ALP': 80, 'ALT': 25, 'AST': 20, 'BIL': 0.7, 'CHE': 9.0, 'CHOL': 220, 'CREA': 0.9, 'GGT': 30, 'PROT': 7.5},
    {'Age': 65, 'Sex': 'F', 'ALB': 2.8, 'ALP': 200, 'ALT': 120, 'AST': 95, 'BIL': 3.5, 'CHE': 5.2, 'CHOL': 160, 'CREA': 1.8, 'GGT': 180, 'PROT': 6.2}
]

print("📦 Batch Predictions:")
for i, patient in enumerate(test_patients):
    result = server.predict(patient)
    print(f"   Patient {i+1}: {result['prediction']} (confidence: {max(result['probabilities'].values()):.3f})")

# Health check
health = server.health_check()
print(f"🏥 Health Check: {health}")

# Cell 6: MLflow Model Registry
print("📊 MLflow Model Registry:")
print("1. Go to MLflow UI in your workspace")
print("2. Find your experiment")
print("3. Click on the best model run")
print("4. Click 'Register Model'")
print("5. Name: 'liver-disease-classifier'")
print("6. Set stage to 'Production'")

# Cell 7: Deployment Summary
print("🎉 DEPLOYMENT READY!")
print("=" * 30)
print(f"✅ Model: {model_package['metadata']['model_name']}")
print(f"✅ Accuracy: {model_package['metadata']['test_accuracy']:.3f}")
print(f"✅ API: Ready for Databricks Model Serving")
print(f"✅ Monitoring: MLflow integration complete")
print(f"✅ Versioning: Model registry ready")

print("\n🚀 Next Steps:")
print("1. Register model in MLflow Model Registry")
print("2. Create Databricks Model Serving endpoint")
print("3. Test with real API calls")
print("4. Set up monitoring and alerts")
print("5. Ready for production!")

print("\n💡 Perfect for Nedbank interview demonstration!")
