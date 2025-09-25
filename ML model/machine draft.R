# Student Performance Analysis
# DS7003 Project Script
# u123456

# Load required packages
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(xgboost)
library(pROC)

# Seed for reproducibility
set.seed(123)

# 1. Data Loading & Preprocessing
# -------------------------------------------------
# Download dataset from UCI and load
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
download.file(url, "student.zip")
unzip("student.zip")
math_data <- read.csv2("student-mat.csv", sep = ";")

# Convert target variable: Pass (G3 >= 10) / Fail
math_data <- math_data %>%
  mutate(pass = as.factor(ifelse(G3 >= 10, "pass", "fail")),
         across(c(school, sex, address, famsize, Pstatus, schoolsup,
                  famsup, paid, activities, nursery, higher, internet,
                  romantic, Mjob, Fjob, reason, guardian), as.factor)) %>%
  select(-G3)  # Remove original grade for classification

# 2. Data Splitting
# -------------------------------------------------
# Classification task
train_index <- createDataPartition(math_data$pass, p = 0.8, list = FALSE)
train_cls <- math_data[train_index, ]
test_cls <- math_data[-train_index, ]

# Regression task (using original G3 values)
math_reg <- read.csv2("student-mat.csv", sep = ";") %>%
  mutate(across(c(school, sex, address, famsize, Pstatus, schoolsup,
                  famsup, paid, activities, nursery, higher, internet,
                  romantic, Mjob, Fjob, reason, guardian), as.factor))

train_index_reg <- createDataPartition(math_reg$G3, p = 0.8, list = FALSE)
train_reg <- math_reg[train_index_reg, ]
test_reg <- math_reg[-train_index_reg, ]

# 3. Preprocessing Pipeline
# -------------------------------------------------
preprocessor <- preProcess(
  train_cls,
  method = c("center", "scale", "nzv")
)

train_cls_processed <- predict(preprocessor, train_cls)
test_cls_processed <- predict(preprocessor, test_cls)

train_reg_processed <- predict(preprocessor, train_reg)
test_reg_processed <- predict(preprocessor, test_reg)

# 4. Classification Models
# -------------------------------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Logistic Regression
logit_model <- train(
  pass ~ .,
  data = train_cls_processed,
  method = "glm",
  family = "binomial",
  trControl = ctrl
)

# Decision Tree
tree_model <- train(
  pass ~ .,
  data = train_cls_processed,
  method = "rpart",
  trControl = ctrl,
  tuneLength = 10
)

# Random Forest
rf_model <- train(
  pass ~ .,
  data = train_cls_processed,
  method = "rf",
  trControl = ctrl,
  ntree = 100
)

# SVM
svm_model <- train(
  pass ~ .,
  data = train_cls_processed,
  method = "svmRadial",
  trControl = ctrl,
  tuneLength = 5
)

# 5. Regression Models
# -------------------------------------------------
# Linear Regression
lm_model <- train(
  G3 ~ .,
  data = train_reg_processed,
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

# XGBoost
xgb_model <- train(
  G3 ~ .,
  data = train_reg_processed,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 3
)

# 6. Model Evaluation
# -------------------------------------------------
# Classification Evaluation
classification_results <- resamples(
  list(
    Logistic = logit_model,
    Tree = tree_model,
    RF = rf_model,
    SVM = svm_model
  )
)

# Regression Evaluation
regression_results <- resamples(
  list(
    Linear = lm_model,
    XGBoost = xgb_model
  )
)

# 7. Output Results
# -------------------------------------------------
# Classification Metrics
print("Classification Results:")
summary(classification_results)

# ROC Curve for best classifier
prob <- predict(rf_model, test_cls_processed, type = "prob")
roc_obj <- roc(test_cls_processed$pass, prob$pass)
plot(roc_obj, main = "ROC Curve")

# Regression Metrics
print("Regression Results:")
summary(regression_results)

# Variable Importance
varImp(rf_model)
varImp(xgb_model)

