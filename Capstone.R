# --------------------------------------------------
# STEP 1: Load Required Libraries
# --------------------------------------------------
library(tidyverse)    # Data manipulation & visualization
library(caret)        # Machine learning framework
library(pROC)         # ROC curve analysis
library(randomForest) # Random Forest implementation
library(ROSE)         # For handling class imbalance
library(glmnet)       # Regularized logistic regression

# Justification: Core packages for data manipulation (tidyverse),
# machine learning (caret, randomForest), class imbalance handling (ROSE),
# and regularized regression (glmnet)

# --------------------------------------------------
# STEP 2: Data Loading & Preprocessing (Verified)
# --------------------------------------------------

# Load dataset with proper factor conversion
creditcard <- read.csv("BankChurners.csv", 
                       stringsAsFactors = TRUE) %>%  # Auto-convert strings to factors
  select(-CLIENTNUM) %>%  # Remove unique identifier
  select(-contains("X"))  # Remove system-generated columns

# Verify raw Attrition_Flag values
print("Original Attrition_Flag values:")
print(levels(creditcard$Attrition_Flag))  # Should show text labels

# Convert target variable with proper level mapping
creditcard <- creditcard %>%
  mutate(Attrition_Flag = factor(
    ifelse(Attrition_Flag == "Existing Customer", 0, 1),
    levels = c(0, 1),
    labels = c("Retained", "Churned")
  ))

# Check class distribution
print("Class distribution after conversion:")
print(table(creditcard$Attrition_Flag))

# Handle missing values
creditcard <- creditcard %>%
  mutate_if(is.numeric, ~ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%  # Numeric: median
  mutate_if(is.factor, ~fct_explicit_na(., na_level = "Missing"))  # Categorical: new level

# Justification: 
# - Proper factor handling from CSV read
# - Explicit NA handling for both numeric and categorical
# - Target variable conversion with level verification

# --------------------------------------------------
# STEP 3: Class Imbalance Handling
# --------------------------------------------------

# Check initial imbalance
class_dist <- prop.table(table(creditcard$Attrition_Flag))
print(paste("Class balance (Retained:Churned):", 
            round(class_dist[1], 2), ":", 
            round(class_dist[2], 2)))

# Apply ROSE sampling for balanced data
set.seed(42)
creditcard_balanced <- ROSE(Attrition_Flag ~ ., 
                            data = creditcard,
                            p = 0.5)$data

# Verify new distribution
print("Balanced class distribution:")
print(table(creditcard_balanced$Attrition_Flag))

# Justification: ROSE (Random Over-Sampling Examples) creates 
# synthetic samples for minority class while maintaining 
# original data distribution characteristics

# --------------------------------------------------
# STEP 4: Train-Test Split with Stratification
# --------------------------------------------------

set.seed(42)
train_index <- createDataPartition(creditcard_balanced$Attrition_Flag,
                                   p = 0.7,
                                   list = FALSE)
train_data <- creditcard_balanced[train_index, ]
test_data <- creditcard_balanced[-train_index, ]

# Verify split integrity
print("Training set dimensions:")
print(dim(train_data))
print("Test set dimensions:")
print(dim(test_data))

print("Training class distribution:")
print(prop.table(table(train_data$Attrition_Flag)))

# Justification: Stratified sampling ensures proportional
# representation of both classes in train/test sets

# --------------------------------------------------
# STEP 5: Model Training with Regularization
# --------------------------------------------------

# 5.1 Regularized Logistic Regression
x_train <- model.matrix(Attrition_Flag ~ .-1, data = train_data)
y_train <- train_data$Attrition_Flag

# Cross-validated LASSO regression
cv_lasso <- cv.glmnet(x_train, y_train,
                      family = "binomial",
                      alpha = 1,
                      type.measure = "auc")

# Final model
logistic_model <- glmnet(x_train, y_train,
                         family = "binomial",
                         alpha = 1,
                         lambda = cv_lasso$lambda.min)

# 5.2 Balanced Random Forest
rf_model <- randomForest(
  Attrition_Flag ~ .,
  data = train_data,
  ntree = 500,
  sampsize = c("Retained" = 1000, "Churned" = 1000),  # Balanced sampling
  strata = train_data$Attrition_Flag,
  importance = TRUE,
  mtry = floor(sqrt(ncol(train_data) - 1))
)

# Justification:
# - LASSO regularization prevents overfitting in logistic regression
# - Balanced RF parameters: explicit class weights, adequate trees,
#   and proper variable sampling (mtry)

# --------------------------------------------------
# STEP 6: Model Evaluation
# --------------------------------------------------

# Prepare test data matrix
x_test <- model.matrix(Attrition_Flag ~ .-1, data = test_data)
y_test <- test_data$Attrition_Flag

# 6.1 Logistic Regression Evaluation
logistic_pred <- predict(logistic_model, 
                         newx = x_test,
                         type = "response")[,1]
logistic_roc <- roc(y_test, logistic_pred)

# 6.2 Random Forest Evaluation
rf_pred <- predict(rf_model, 
                   newdata = test_data,
                   type = "prob")[,"Churned"]
rf_roc <- roc(y_test, rf_pred)

# Compare performance
model_performance <- data.frame(
  Model = c("Regularized Logistic", "Balanced Random Forest"),
  AUC = c(auc(logistic_roc), auc(rf_roc)),
  Sensitivity = c(coords(logistic_roc, "best", ret = "sensitivity")[1],
                  coords(rf_roc, "best", ret = "sensitivity")[1]),
  Specificity = c(coords(logistic_roc, "best", ret = "specificity")[1],
                  coords(rf_roc, "best", ret = "specificity")[1])
)

print("Model Performance Comparison:")
print(model_performance)

# Justification: Comprehensive evaluation using AUC and 
# optimal threshold-based sensitivity/specificity

# --------------------------------------------------
# STEP 7: Feature Importance Analysis
# --------------------------------------------------

# Logistic Regression Coefficients
logistic_coef <- coef(logistic_model) %>%
  as.matrix() %>%
  data.frame() %>%
  arrange(-abs(.))

# Random Forest Importance
rf_importance <- importance(rf_model) %>%
  data.frame() %>%
  arrange(-MeanDecreaseGini)

print("Top 10 Logistic Regression Features:")
print(head(logistic_coef, 10))

print("Top 10 Random Forest Features:")
print(head(rf_importance, 10))

# Justification: Dual feature importance perspective -
# regularization-based (LASSO) and tree-based (RF)

# --------------------------------------------------
# STEP 8: Deployment Preparation
# --------------------------------------------------

# Save final model objects
saveRDS(logistic_model, "churn_logistic_model.rds")
saveRDS(rf_model, "churn_rf_model.rds")

# Save preprocessing parameters
preprocessing_params <- list(
  median_values = sapply(creditcard[numeric_cols], median, na.rm = TRUE),
  factor_levels = lapply(creditcard[categorical_cols], levels)
)

saveRDS(preprocessing_params, "preprocessing_params.rds")

# Justification: Persisting models and preprocessing parameters
# ensures reproducibility in deployment