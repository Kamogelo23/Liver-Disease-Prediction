# Liver-Disease-Prediction
This repository contains an R script for analyzing a dataset related to Hepatitis C Virus (HCV). The script performs a comprehensive analysis including:

Data Loading and Exploration: Reads the dataset "hcvdat0.csv" containing 615 observations and 14 variables, including a response variable with four levels (Blood donor, Cirrhosis, Fibrosis, Hepatitis).

Package Installation and Library Loading: Installs and loads various packages required for data analysis, visualization, and modeling.

# Data Cleaning: Identifies and handles missing values through imputation using predictive mean matching, and normalizes features using Z-score normalization. Outliers are also handled using Cook's distance.

# Exploratory Data Analysis (EDA): Includes histograms, box plots, and summary statistics to understand the distribution and characteristics of the data.

Variable Selection: Performs Principal Component Analysis (PCA) and feature plotting to select significant variables.

Model Fitting: Trains several models including Linear Discriminant Analysis (LDA), Decision Trees (CART), k-Nearest Neighbors (k-NN), Support Vector Machines (SVM), and Random Forest (RF) using cross-validation. The models are evaluated, compared, and validated using accuracy as the metric.

Visualization: Includes various plots to visualize the data and model performance.
