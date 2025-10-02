# Liver Disease Prediction - Professional ML Pipeline

A comprehensive machine learning pipeline for liver disease prediction using Hepatitis C Virus (HCV) data. This project provides both **R** and **Python** implementations with professional-grade code structure, extensive documentation, and modern ML best practices.

## 🚀 Features

- **Dual Language Support**: Complete implementations in both R and Python
- **Professional Code Structure**: Well-documented, modular, and maintainable code
- **Comprehensive ML Pipeline**: End-to-end analysis from data loading to model deployment
- **Advanced Preprocessing**: Missing value imputation, outlier detection, and feature scaling
- **Multiple ML Models**: Random Forest, SVM, Decision Trees, k-NN, LDA, and more
- **Class Imbalance Handling**: SMOTE implementation for balanced training
- **Rich Visualizations**: Professional plots and interactive dashboards
- **Model Evaluation**: Comprehensive metrics and cross-validation
- **Sample Dataset**: Realistic synthetic data generator for testing

## 📊 Dataset Information

The dataset contains **615 observations** with **14 variables** including:
- **Demographics**: Age, Sex
- **Liver Function Markers**: ALB (Albumin), ALP (Alkaline Phosphatase), ALT (Alanine Aminotransferase), AST (Aspartate Aminotransferase), BIL (Bilirubin)
- **Other Biomarkers**: CHE (Cholinesterase), CHOL (Cholesterol), CREA (Creatinine), GGT (Gamma-glutamyl transferase), PROT (Protein)
- **Target Variable**: Category (Blood donor, Cirrhosis, Fibrosis, Hepatitis)

## 🛠️ Installation

### Python Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Liver-Disease-Prediction
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Generate sample dataset** (if you don't have the original data):
```bash
python generate_sample_data.py
```

### R Setup

1. **Install required R packages**:
```r
# The R script will automatically install missing packages
source("liver_disease_prediction_R.R")
```

## 🚀 Quick Start

### Python Implementation

```bash
# Run the complete pipeline
python liver_disease_prediction_python.py

# Or use as a module
python -c "
from liver_disease_prediction_python import LiverDiseasePredictor
predictor = LiverDiseasePredictor()
predictor.load_data()
predictor.explore_data()
predictor.preprocess_data()
predictor.split_data()
predictor.train_models()
predictor.evaluate_model()
"
```

### R Implementation

```r
# Source the complete R script
source("liver_disease_prediction_R.R")
```

## 📈 Key Improvements Made

### Code Quality
- ✅ **Professional Documentation**: Comprehensive docstrings and comments
- ✅ **Error Handling**: Robust error handling and logging
- ✅ **Modular Design**: Object-oriented Python implementation
- ✅ **Code Organization**: Clear separation of concerns and functions

### Data Processing
- ✅ **Advanced Imputation**: KNN imputation for missing values
- ✅ **Outlier Detection**: Cook's distance and statistical methods
- ✅ **Feature Scaling**: StandardScaler for consistent preprocessing
- ✅ **Data Validation**: Comprehensive data quality checks

### Machine Learning
- ✅ **Multiple Models**: 7 different algorithms for comparison
- ✅ **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- ✅ **Cross-Validation**: 10-fold CV for robust evaluation
- ✅ **Class Balancing**: SMOTE for handling imbalanced classes
- ✅ **Feature Importance**: Analysis of key predictive features

### Visualization
- ✅ **Professional Plots**: High-quality matplotlib and seaborn visualizations
- ✅ **Interactive Dashboards**: Plotly integration for exploration
- ✅ **Comprehensive EDA**: Distribution plots, correlation matrices, box plots
- ✅ **Model Performance**: Confusion matrices, ROC curves, feature importance

### Evaluation Metrics
- ✅ **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score
- ✅ **Classification Reports**: Detailed per-class performance
- ✅ **Confusion Matrices**: Visual representation of predictions
- ✅ **Cross-Validation**: Robust performance estimation

## 📁 Project Structure

```
Liver-Disease-Prediction/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                           # Python package setup
├── liver_disease_prediction_python.py # Main Python script
├── liver_disease_prediction_R.R       # Improved R script
├── generate_sample_data.py            # Dataset generator
├── R codes                            # Original R code (for reference)
└── output/                            # Generated outputs
    ├── eda_analysis.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── best_liver_disease_model.pkl
    └── liver_disease_prediction.log
```

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Imputation**: KNN imputation for realistic data handling
2. **Outlier Detection**: Statistical methods to identify and handle outliers
3. **Feature Scaling**: Standardization for consistent model performance
4. **Categorical Encoding**: Proper handling of categorical variables

### Model Training
1. **Data Splitting**: 80/20 train-test split with stratification
2. **Class Balancing**: SMOTE to handle class imbalance
3. **Cross-Validation**: 10-fold CV for robust model evaluation
4. **Multiple Algorithms**: Comparison of 7 different ML approaches

### Model Evaluation
1. **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
2. **Confusion Matrices**: Detailed prediction analysis
3. **Feature Importance**: Understanding key predictive factors
4. **ROC Analysis**: Model discrimination capability

## 📋 Enhanced Logging System

### Python Logging Features
- **Multi-level Logging**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Structured JSON Logging**: Machine-readable logs for analysis
- **Performance Tracking**: Automatic timing of all major operations
- **Multiple Output Formats**: Console, file, and structured JSON
- **Comprehensive Data Profiling**: Automatic logging of dataset characteristics
- **Model Performance Metrics**: Detailed logging of training and evaluation metrics

### R Logging Features
- **Timestamped Messages**: All log entries include precise timestamps
- **Performance Monitoring**: Track execution time for each major step
- **Structured Output**: Consistent formatting across all log messages
- **Error Handling**: Comprehensive error logging with context
- **Progress Tracking**: Clear indication of pipeline progress

### Log Files Generated
- `liver_disease_prediction_YYYYMMDD_HHMMSS.log`: Detailed human-readable log
- `liver_disease_prediction_structured.json`: Machine-readable structured log
- `errors.log`: Error-specific log file (Python only)

### Logging Configuration
- **Customizable**: Easy to modify log levels and output formats
- **Configurable**: JSON configuration file for Python logging
- **Flexible**: Support for different environments (development, production)

## 📊 Expected Results

The improved pipeline typically achieves:
- **Accuracy**: 85-95% depending on data quality
- **Best Performing Models**: Random Forest and Gradient Boosting
- **Key Features**: ALT, AST, ALB, and BIL are typically most important
- **Class Performance**: Balanced performance across all disease categories

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the logs in the `output/` directory
2. Review the documentation in the code
3. Open an issue on GitHub

## 🙏 Acknowledgments

- Original R implementation for the foundational analysis approach
- Medical literature for realistic biomarker ranges
- Scikit-learn and R caret communities for excellent ML tools
