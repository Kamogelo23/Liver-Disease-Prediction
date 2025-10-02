# =============================================================================
# Liver Disease Prediction - Professional R Script
# =============================================================================
# Author: Improved version of original project
# Date: 2024
# Description: Comprehensive analysis and prediction of liver disease using HCV data
# =============================================================================

# Clear workspace
rm(list = ls())
gc()

# =============================================================================
# LOGGING AND UTILITY FUNCTIONS
# =============================================================================

# Create log file with timestamp
log_file <- paste0("liver_disease_prediction_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log")

# Function to log messages with timestamp
log_message <- function(message, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  log_entry <- paste0(timestamp, " | ", level, " | ", message)
  
  # Print to console
  cat(log_entry, "\n")
  
  # Write to log file
  if (!file.exists("output")) {
    dir.create("output", showWarnings = FALSE)
  }
  write(log_entry, file = file.path("output", log_file), append = TRUE)
}

# Function to log performance metrics
log_performance <- function(step_name, start_time, end_time = Sys.time()) {
  duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
  log_message(paste0("PERFORMANCE | ", step_name, " completed in ", round(duration, 2), " seconds"))
  return(duration)
}

# Function to log data information
log_data_info <- function(data, name = "Dataset") {
  log_message(paste0("=== ", toupper(name), " INFORMATION ==="))
  log_message(paste0("Shape: ", nrow(data), " rows x ", ncol(data), " columns"))
  log_message(paste0("Memory usage: ", round(object.size(data) / 1024^2, 2), " MB"))
  
  # Missing values
  missing_values <- sapply(data, function(x) sum(is.na(x)))
  total_missing <- sum(missing_values)
  if (total_missing > 0) {
    log_message(paste0("Missing values: ", total_missing, " (", round(total_missing/prod(dim(data))*100, 2), "%)"), "WARNING")
    for (col in names(missing_values[missing_values > 0])) {
      log_message(paste0("  ", col, ": ", missing_values[col], " (", round(missing_values[col]/nrow(data)*100, 2), "%)"), "WARNING")
    }
  } else {
    log_message("No missing values found")
  }
  
  # Data types
  numeric_cols <- sapply(data, is.numeric)
  categorical_cols <- sapply(data, function(x) is.factor(x) || is.character(x))
  log_message(paste0("Numeric variables: ", sum(numeric_cols)))
  log_message(paste0("Categorical variables: ", sum(categorical_cols)))
}

# Function to log model performance
log_model_performance <- function(model_name, metrics, execution_time) {
  log_message(paste0("=== ", toupper(model_name), " PERFORMANCE ==="))
  for (metric in names(metrics)) {
    log_message(paste0(metric, ": ", round(metrics[[metric]], 4)))
  }
  log_message(paste0("Training time: ", round(execution_time, 2), " seconds"))
}

# Initialize logging
log_message("================================================================================")
log_message("LIVER DISEASE PREDICTION PIPELINE STARTED")
log_message("================================================================================")
log_message(paste0("R version: ", R.version.string))
log_message(paste0("Working directory: ", getwd()))
log_message(paste0("Log file: ", log_file))
log_message(paste0("Start time: ", Sys.time()))

# Track overall performance
pipeline_start_time <- Sys.time()
performance_metrics <- list(
  data_loading_time = 0,
  preprocessing_time = 0,
  training_time = 0,
  evaluation_time = 0,
  total_time = 0
)

# =============================================================================
# 1. PACKAGE INSTALLATION AND LOADING
# =============================================================================

# Function to install and load packages
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# Required packages
required_packages <- c(
  "ggplot2", "dplyr", "caret", "randomForest", "e1071", "kernlab",
  "rpart", "rpart.plot", "RColorBrewer", "VIM", "mice", "corrplot",
  "pROC", "plotROC", "gridExtra", "factoextra", "DMwR"
)

# Install and load packages
install_and_load(required_packages)

# =============================================================================
# 2. DATA LOADING AND VALIDATION
# =============================================================================

# Function to load data with error handling
load_data <- function(file_path) {
  if (!file.exists(file_path)) {
    cat("Warning: Dataset file not found. Creating sample dataset...\n")
    return(create_sample_dataset())
  }
  
  tryCatch({
    data <- read.csv(file_path, stringsAsFactors = FALSE)
    cat("Dataset loaded successfully!\n")
    cat("Dimensions:", dim(data), "\n")
    return(data)
  }, error = function(e) {
    cat("Error loading data:", e$message, "\n")
    cat("Creating sample dataset instead...\n")
    return(create_sample_dataset())
  })
}

# Function to create sample dataset for demonstration
create_sample_dataset <- function() {
  set.seed(123)
  n <- 615
  
  # Create synthetic HCV-like data
  data <- data.frame(
    Age = round(rnorm(n, mean = 45, sd = 15)),
    Sex = sample(c("m", "f"), n, replace = TRUE),
    ALB = round(rnorm(n, mean = 4.2, sd = 0.6), 2),
    ALP = round(rnorm(n, mean = 95, sd = 35)),
    ALT = round(rnorm(n, mean = 35, sd = 25)),
    AST = round(rnorm(n, mean = 30, sd = 20)),
    BIL = round(rnorm(n, mean = 0.8, sd = 0.4), 2),
    CHE = round(rnorm(n, mean = 8.5, sd = 2.0), 2),
    CHOL = round(rnorm(n, mean = 200, sd = 50)),
    CREA = round(rnorm(n, mean = 1.0, sd = 0.3), 2),
    GGT = round(rnorm(n, mean = 40, sd = 30)),
    PROT = round(rnorm(n, mean = 7.0, sd = 0.8), 2),
    Category = sample(c("Blood donor", "Cirrhosis", "Fibrosis", "Hepatitis"), 
                     n, replace = TRUE, prob = c(0.4, 0.2, 0.2, 0.2))
  )
  
  # Add some missing values
  missing_indices <- sample(1:n, 31)
  data$ALB[missing_indices[1:10]] <- NA
  data$ALP[missing_indices[11:20]] <- NA
  data$BIL[missing_indices[21:31]] <- NA
  
  cat("Sample dataset created with", n, "observations and", ncol(data), "variables\n")
  return(data)
}

# Load the data
log_message("Starting data loading process...")
data_loading_start <- Sys.time()
data <- load_data("hcvdat0.csv")
performance_metrics$data_loading_time <- log_performance("Data Loading", data_loading_start)

# =============================================================================
# 3. DATA EXPLORATION AND STRUCTURE
# =============================================================================

log_message("Starting data exploration...")
log_data_info(data, "Original Dataset")

# Check for missing values
missing_summary <- sapply(data, function(x) sum(is.na(x)))
log_message("Missing values summary:")
for (col in names(missing_summary)) {
  if (missing_summary[col] > 0) {
    log_message(paste0("  ", col, ": ", missing_summary[col]), "WARNING")
  }
}

# Visualize missing data pattern
if (sum(missing_summary) > 0) {
  png("missing_data_pattern.png", width = 800, height = 600)
  aggr(data, col = c("navyblue", "yellow"), numbers = TRUE, sortVars = TRUE,
       labels = names(data), cex.axis = 0.7, gap = 3, 
       ylab = c("Missing Data", "Pattern"))
  dev.off()
}

# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================

log_message("Starting data preprocessing...")
preprocessing_start <- Sys.time()

# Function for data preprocessing
preprocess_data <- function(df) {
  log_message("Converting categorical variables...")
  # Convert categorical variables
  df$Sex <- as.factor(df$Sex)
  df$Category <- as.factor(df$Category)
  
  log_message("Creating dummy variables...")
  # Create dummy variables for categorical features
  dummy_vars <- dummyVars("~ Sex", data = df, fullRank = TRUE)
  sex_dummies <- data.frame(predict(dummy_vars, newdata = df))
  
  # Combine numeric and dummy variables
  numeric_vars <- df[, !names(df) %in% c("Sex", "Category")]
  processed_df <- cbind(numeric_vars, sex_dummies, Category = df$Category)
  
  log_message(paste0("Preprocessed data shape: ", nrow(processed_df), " x ", ncol(processed_df)))
  return(processed_df)
}

# Preprocess the data
processed_data <- preprocess_data(data)

# Handle missing values using MICE imputation
if (sum(is.na(processed_data)) > 0) {
  cat("Performing MICE imputation...\n")
  imputed_data <- mice(processed_data, m = 5, maxit = 50, method = "pmm", seed = 123)
  final_data <- complete(imputed_data, 1)
} else {
  final_data <- processed_data
}

# Remove outliers using Cook's distance (for continuous variables)
numeric_cols <- sapply(final_data, is.numeric)
numeric_data <- final_data[, numeric_cols & names(final_data) != "Category"]

if (ncol(numeric_data) > 0) {
  # Fit a linear model to identify outliers
  lm_model <- lm(Category ~ ., data = final_data[, c(numeric_data, "Category")])
  cooks_dist <- cooks.distance(lm_model)
  
  # Remove outliers (Cook's distance > 4*mean)
  outlier_threshold <- 4 * mean(cooks_dist, na.rm = TRUE)
  outliers <- which(cooks_dist > outlier_threshold)
  
  if (length(outliers) > 0) {
    cat("Removing", length(outliers), "outliers based on Cook's distance\n")
    final_data <- final_data[-outliers, ]
  }
}

# Normalize numeric features (Z-score normalization)
normalize_features <- function(df) {
  numeric_cols <- sapply(df, is.numeric)
  df[, numeric_cols] <- scale(df[, numeric_cols])
  return(df)
}

normalized_data <- normalize_features(final_data)

cat("Final dataset dimensions:", dim(normalized_data), "\n")

# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS
# =============================================================================

cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# Create comprehensive EDA plots
create_eda_plots <- function(df) {
  # Distribution plots
  p1 <- ggplot(df, aes(x = Category, fill = Category)) +
    geom_bar() +
    labs(title = "Distribution of Disease Categories", 
         x = "Category", y = "Count") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Box plots for key variables
  key_vars <- c("Age", "ALB", "ALT", "AST", "BIL")
  available_vars <- key_vars[key_vars %in% names(df)]
  
  if (length(available_vars) > 0) {
    plot_list <- list()
    for (var in available_vars) {
      plot_list[[var]] <- ggplot(df, aes_string(x = "Category", y = var, fill = "Category")) +
        geom_boxplot() +
        labs(title = paste("Distribution of", var, "by Category"),
             x = "Category", y = var) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }
    
    # Save plots
    png("eda_distribution.png", width = 1000, height = 600)
    print(p1)
    dev.off()
    
    png("eda_boxplots.png", width = 1200, height = 800)
    do.call(grid.arrange, c(plot_list, ncol = 2))
    dev.off()
  }
}

create_eda_plots(final_data)

# Correlation analysis
numeric_cols <- sapply(final_data, is.numeric)
if (sum(numeric_cols) > 1) {
  cor_matrix <- cor(final_data[, numeric_cols], use = "complete.obs")
  
  png("correlation_matrix.png", width = 800, height = 800)
  corrplot(cor_matrix, method = "color", type = "upper", 
           addCoef.col = "black", tl.cex = 0.8, number.cex = 0.6)
  dev.off()
}

# =============================================================================
# 6. FEATURE SELECTION
# =============================================================================

cat("\n=== FEATURE SELECTION ===\n")

# Principal Component Analysis
perform_pca <- function(df) {
  numeric_cols <- sapply(df, is.numeric)
  pca_data <- df[, numeric_cols]
  
  if (ncol(pca_data) > 1) {
    pca_result <- prcomp(pca_data, center = TRUE, scale. = TRUE)
    
    # Plot PCA results
    png("pca_variance_explained.png", width = 800, height = 600)
    fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50))
    dev.off()
    
    return(pca_result)
  }
  return(NULL)
}

pca_result <- perform_pca(final_data)

# =============================================================================
# 7. MODEL TRAINING AND EVALUATION
# =============================================================================

log_message("Starting model training process...")
training_start <- Sys.time()

# Prepare data for modeling
log_message("Splitting data into training and test sets...")
set.seed(123)
train_indices <- createDataPartition(normalized_data$Category, p = 0.8, list = FALSE)
train_data <- normalized_data[train_indices, ]
test_data <- normalized_data[-train_indices, ]
log_message(paste0("Training set: ", nrow(train_data), " samples"))
log_message(paste0("Test set: ", nrow(test_data), " samples"))

# Handle class imbalance using SMOTE
if ("DMwR" %in% loadedNamespaces()) {
  tryCatch({
    balanced_data <- SMOTE(Category ~ ., train_data, perc.over = 100, k = 5)
    cat("SMOTE applied to balance classes\n")
  }, error = function(e) {
    cat("SMOTE failed, using original data:", e$message, "\n")
    balanced_data <- train_data
  })
} else {
  balanced_data <- train_data
}

# Define training control
train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

# Train multiple models
models <- list()

# 1. Random Forest
log_message("Training Random Forest...")
model_start <- Sys.time()
set.seed(123)
models$rf <- train(
  Category ~ ., 
  data = balanced_data, 
  method = "rf", 
  trControl = train_control,
  metric = "Accuracy",
  importance = TRUE
)
model_time <- log_performance("Random Forest Training", model_start)
log_model_performance("Random Forest", 
                     list(CV_Accuracy = max(models$rf$results$Accuracy)), 
                     model_time)

# 2. Support Vector Machine
log_message("Training SVM...")
model_start <- Sys.time()
set.seed(123)
models$svm <- train(
  Category ~ ., 
  data = balanced_data, 
  method = "svmRadial", 
  trControl = train_control,
  metric = "Accuracy"
)
model_time <- log_performance("SVM Training", model_start)
log_model_performance("SVM", 
                     list(CV_Accuracy = max(models$svm$results$Accuracy)), 
                     model_time)

# 3. Decision Tree
log_message("Training Decision Tree...")
model_start <- Sys.time()
set.seed(123)
models$cart <- train(
  Category ~ ., 
  data = balanced_data, 
  method = "rpart", 
  trControl = train_control,
  metric = "Accuracy"
)
model_time <- log_performance("Decision Tree Training", model_start)
log_model_performance("Decision Tree", 
                     list(CV_Accuracy = max(models$cart$results$Accuracy)), 
                     model_time)

# 4. k-Nearest Neighbors
log_message("Training k-NN...")
model_start <- Sys.time()
set.seed(123)
models$knn <- train(
  Category ~ ., 
  data = balanced_data, 
  method = "knn", 
  trControl = train_control,
  metric = "Accuracy"
)
model_time <- log_performance("k-NN Training", model_start)
log_model_performance("k-NN", 
                     list(CV_Accuracy = max(models$knn$results$Accuracy)), 
                     model_time)

# 5. Linear Discriminant Analysis
log_message("Training LDA...")
model_start <- Sys.time()
set.seed(123)
models$lda <- train(
  Category ~ ., 
  data = balanced_data, 
  method = "lda", 
  trControl = train_control,
  metric = "Accuracy"
)
model_time <- log_performance("LDA Training", model_start)
log_model_performance("LDA", 
                     list(CV_Accuracy = max(models$lda$results$Accuracy)), 
                     model_time)

# =============================================================================
# 8. MODEL EVALUATION
# =============================================================================

log_message("Starting model evaluation...")
evaluation_start <- Sys.time()

# Compare model performance
log_message("Comparing model performance...")
results <- resamples(models)
summary(results)

# Create comparison plot
log_message("Creating model comparison plot...")
png("output/model_comparison.png", width = 1000, height = 600)
dotplot(results, main = "Model Performance Comparison")
dev.off()

# Get best model
best_model_name <- names(models)[which.max(sapply(models, function(x) max(x$results$Accuracy, na.rm = TRUE)))]
best_model <- models[[best_model_name]]
log_message(paste0("Best model: ", best_model_name), "SUCCESS")

# Make predictions on test set
log_message("Making predictions on test set...")
predictions <- predict(best_model, test_data)
confusion_matrix <- confusionMatrix(predictions, test_data$Category)
log_message("Confusion Matrix Results:")
log_message(paste0("Overall Accuracy: ", round(confusion_matrix$overall["Accuracy"], 4)))
log_message(paste0("Kappa: ", round(confusion_matrix$overall["Kappa"], 4)))

# Plot confusion matrix
png("confusion_matrix.png", width = 800, height = 600)
plot(confusion_matrix$table, main = "Confusion Matrix")
dev.off()

# Feature importance (for tree-based models)
if (inherits(best_model, "train") && best_model$method == "rf") {
  importance <- varImp(best_model)
  png("feature_importance.png", width = 800, height = 600)
  plot(importance, main = "Feature Importance (Random Forest)")
  dev.off()
}

# =============================================================================
# 9. ROC ANALYSIS (for binary classification)
# =============================================================================

# Create ROC curves for each class (one-vs-rest)
create_roc_curves <- function(model, test_data) {
  if (length(unique(test_data$Category)) == 2) {
    predictions_proba <- predict(model, test_data, type = "prob")
    roc_curve <- roc(test_data$Category, predictions_proba[, 2])
    
    png("roc_curve.png", width = 800, height = 600)
    plot(roc_curve, main = "ROC Curve", print.auc = TRUE)
    dev.off()
  }
}

create_roc_curves(best_model, test_data)

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================

cat("\n=== SAVING RESULTS ===\n")

# Save the best model
saveRDS(best_model, "best_liver_disease_model.rds")

# Save predictions
results_df <- data.frame(
  Actual = test_data$Category,
  Predicted = predictions,
  Model = rep(names(models)[which.max(sapply(models, function(x) max(x$results$Accuracy, na.rm = TRUE)))], 
              length(predictions))
)
write.csv(results_df, "predictions_results.csv", row.names = FALSE)

# Update performance metrics
performance_metrics$preprocessing_time <- log_performance("Data Preprocessing", preprocessing_start)
performance_metrics$training_time <- log_performance("Model Training", training_start)
performance_metrics$evaluation_time <- log_performance("Model Evaluation", evaluation_start)
performance_metrics$total_time <- log_performance("Total Pipeline", pipeline_start_time)

# Create summary report
log_message("================================================================================")
log_message("SUMMARY REPORT")
log_message("================================================================================")
log_message("Dataset: HCV Liver Disease Prediction")
log_message(paste0("Total observations: ", nrow(data)))
log_message(paste0("Total features: ", ncol(data)))
log_message(paste0("Missing values handled: ", sum(missing_summary)))
log_message(paste0("Best model: ", best_model_name))
log_message(paste0("Best model CV accuracy: ", round(max(sapply(models, function(x) max(x$results$Accuracy, na.rm = TRUE))), 4)))
log_message(paste0("Test set accuracy: ", round(confusion_matrix$overall["Accuracy"], 4)))

log_message("PERFORMANCE METRICS:")
for (metric in names(performance_metrics)) {
  log_message(paste0("  ", metric, ": ", round(performance_metrics[[metric]], 2), " seconds"))
}

log_message("================================================================================")
log_message("ANALYSIS COMPLETED SUCCESSFULLY!")
log_message("================================================================================")
log_message(paste0("All plots and models saved to: ", getwd(), "/output/"))
log_message(paste0("Log file: ", log_file))
log_message(paste0("End time: ", Sys.time()))
