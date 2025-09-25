# Load required libraries
library(corrplot)
library(tidyverse)      # data manipulation and plotting
library(randomForest)   # random forest models
library(nnet)           # neural networks
library(caret)          # evaluation and splitting
library(ggplot2)        # visualizations
library(e1071)          # SVM
library(class)          # k-NN classification
library(FNN)            # k-NN regression
library(pROC)           # computing and plotting ROC curves and AUC
library(reshape2)       # For reshaping data 

# Set working directory
setwd(dirname(file.choose()))
getwd()

# Load dataset
data <- read.csv("mat.csv")

# Convert character columns to factors
data <- data %>% mutate_if(is.character, as.factor)
str(data)

# Exploratory Plots
# Histogram with density for G3
ggplot(data, aes(x = G3)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "skyblue", color = "black") +
  geom_density(color = "red", size = 1) +
  ggtitle("Normal Distribution of Final Grade (G3)") +
  xlab("Final Grade (G3)") +
  ylab("Density")

# Boxplot of reason vs G3
ggplot(data, aes(x = reason, y = G3)) +
  geom_boxplot(fill = "orange") +
  ggtitle("Box Plot: Reason vs G3") +
  xlab("Reason") + ylab("Final Grade (G3)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Make sure to install this package if not already
numeric_vars <- data %>%
  select(G3, G2, absences, age, health) %>%
  mutate_if(is.factor, as.numeric)

# Convert all character/factor variables to numeric to allow correlation computation
cor_data <- data %>%
  mutate_if(is.factor, ~ as.numeric(as.factor(.)))

# Compute correlation matrix
cor_matrix_all <- cor(cor_data, use = "complete.obs")

# Visualize the full correlation matrix
corrplot(cor_matrix_all,
         method = "color", type = "lower",
         tl.col = "black", tl.cex = 0.6, number.cex = 0.6,
         title = "Correlation Matrix: All Variables", mar = c(0, 0, 2, 0),
         addCoef.col = "black", col = colorRampPalette(c("blue", "white", "red"))(200))

# Get correlations with G3
g3_cor <- cor_matrix_all[, "G3"]
g3_cor <- sort(abs(g3_cor), decreasing = TRUE)

# Select top variables correlated with G3 (excluding G3 itself)
top_vars <- names(g3_cor[g3_cor < 1])[1:10]  # Top 10 excluding G3 itself

# Create a focused correlation matrix
hotspot_matrix <- cor_data %>%
  select(all_of(c("G3", top_vars))) %>%
  cor(use = "complete.obs")

# Plot hotspot map
corrplot(hotspot_matrix,
         method = "shade", type = "lower", order = "hclust",
         tl.col = "black", tl.cex = 0.8, number.cex = 0.7,
         title = "Hotspot Mapping: G3 and Top Correlated Variables", mar = c(0, 0, 2, 0),
         addCoef.col = "black", col = colorRampPalette(c("darkblue", "white", "darkred"))(200))


# Scatter plots for top predictors
plot_vars <- c("G2", "absences", "age", "health")
for (var in plot_vars) {
  print(
    ggplot(data, aes_string(x = var, y = "G3")) +
      geom_point() +
      geom_smooth(method = "lm", se = FALSE, color = "red") +
      ggtitle(paste("Scatter Plot:", var, "vs G3")) +
      xlab(var) + ylab("Final Grade (G3)")
  )
}

# Prepare data for Machine learning
set.seed(123)
data_num <- data %>% mutate_if(is.factor, ~ as.numeric(as.factor(.)))
data_num

#_________________
# Regression Model
#_________________

d1 <- createDataPartition(data_num$G3, p = 0.7, list = FALSE) #splitting of data  
train_data <- data_num[d1, ]
test_data <- data_num[-d1, ]

# Random Forest
rf_reg <- randomForest(G3 ~ ., data = train_data, ntree = 100)
rf_preds <- predict(rf_reg, test_data)
rf_rmse <- RMSE(rf_preds, test_data$G3)
rf_r2 <-  R2(rf_preds, test_data$G3)

# Summary for Random Forest
print("Random Forest Summary:")
print(rf_reg)

# Plot feature importance
varImpPlot(rf_reg, main = "Random Forest Variable Importance")

# Q-Q Plot for residuals
qqnorm(rf_preds - test_data$G3, main = "Random Forest Residuals Q-Q Plot")
qqline(rf_preds - test_data$G3, col = "red")


# Neural Network
nn_reg <- nnet(G3 ~ ., data = train_data, size = 5, linout = TRUE, trace = FALSE)
nn_preds <- predict(nn_reg, test_data)
nn_rmse <- RMSE(nn_preds, test_data$G3)
nn_r2 <- R2(nn_preds, test_data$G3)

# Summary for Neural Network
print("Neural Network Summary:")
print(nn_reg)

# Plot actual vs predicted
plot(test_data$G3, nn_preds,
     main = "Neural Network: Actual vs Predicted",
     xlab = "Actual G3", ylab = "Predicted G3", col = "blue", pch = 16)
abline(0, 1, col = "red")

# SVM
svm_reg <- svm(G3 ~ ., data = train_data)
svm_preds <- predict(svm_reg, test_data)
svm_rmse <-  RMSE(svm_preds, test_data$G3)
svm_r2 <-  R2(svm_preds, test_data$G3)

# Summary for SVM
print("SVM Summary:")
print(svm_reg)

# Plot actual vs predicted
plot(test_data$G3, svm_preds,
     main = "SVM: Actual vs Predicted",
     xlab = "Actual G3", ylab = "Predicted G3", col = "green", pch = 16)
abline(0, 1, col = "red")

# k-NN
train_scaled <- scale(train_data)
test_scaled <- scale(test_data, center = attr(train_scaled, "scaled:center"),
                     scale = attr(train_scaled, "scaled:scale"))
knn_preds <- knn.reg(train = train_scaled[, -which(colnames(train_scaled) == "G3")],
                     test = test_scaled[, -which(colnames(test_scaled) == "G3")],
                     y = train_scaled[, "G3"], k = 5)$pred

knn_rmse <- RMSE(knn_preds, test_data$G3)
knn_r2 <-  R2(knn_preds, test_data$G3)

# k-NN does not have a summary method
print("k-NN Summary: No model object; direct prediction only.")

# Plot actual vs predicted
plot(test_data$G3, knn_preds,
     main = "k-NN: Actual vs Predicted",
     xlab = "Actual G3", ylab = "Predicted G3", col = "purple", pch = 16)
abline(0, 1, col = "red")

# Regression Evaluation
reg_results <- data.frame(
  Model = c("Random Forest", "Neural Network", "SVM", "k-NN"),
  RMSE = c(rf_rmse,nn_rmse,svm_rmse,knn_rmse),
  R2 = c(rf_r2 ,nn_r2, svm_r2, knn_r2))

print("Regression Evaluation:", reg_results)
print(reg_results)
cat("Regression Evaluation:\n"); print(reg_results)


# draw a bar-plot for these
results_melted <- melt(reg_results, id.vars = "Model")

ggplot(results_melted, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Model Comparison: RMSE and R2", x = "Model", y = "Metric Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

#print the best model for this
# Identify the best model (lowest RMSE)
best_model_rmse <- reg_results[which.min(reg_results$RMSE), ]
cat("\nBest model based on lowest RMSE:\n")
print(best_model_rmse)

# Identify the best model (highest R2)
best_model_r2 <- reg_results[which.max(reg_results$R2), ]
cat("\nBest model based on highest R²:\n")
print(best_model_r2)

#_________________________
# Classification Task
#_________________________

#Leveling the final grade into Low, Medium and High

G3_class <- cut(data_num$G3, breaks = c(-1, 9, 14, 20), labels = c("Low", "Medium", "High"))
data_num$G3_class <- G3_class

#Splitting data for training and testing

c1 <- createDataPartition(data_num$G3_class, p = 0.7, list = FALSE)
train_cls <- data_num[c1, ]
test_cls <- data_num[-c1, ]
levels_to_use <- levels(train_cls$G3_class)
print(levels_to_use)

# Random Forest
rf_cls <- randomForest(G3_class ~ . -G3, data = train_cls, ntree = 100)
rf_cls_pred <- predict(rf_cls, test_cls)

# Confusion Matrix
rf_cls_pred <- factor(rf_cls_pred, levels = levels_to_use)
test_cls$G3_class <- factor(test_cls$G3_class, levels = levels_to_use)
conf_rf <- confusionMatrix(rf_cls_pred, test_cls$G3_class)
print(conf_rf)

# Confusion Matrix Plot
rf_cm <- table(Predicted = rf_cls_pred, Actual = test_cls$G3_class)
ggplot(as.data.frame(rf_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  ggtitle("Random Forest Confusion Matrix") +
  theme_minimal()

# Neural Network
nn_cls <- nnet(G3_class ~ . -G3, data = train_cls, size = 5, trace = FALSE, maxit = 200)
nn_cls_pred <- predict(nn_cls, test_cls, type = "class")

# Confusion Matrix
nn_cls_pred <- factor(nn_cls_pred, levels = levels_to_use)
conf_nn <- confusionMatrix(nn_cls_pred, test_cls$G3_class)
print(conf_nn)


# Confusion Matrix Plot
nn_cm <- table(Predicted = nn_cls_pred, Actual = test_cls$G3_class)
ggplot(as.data.frame(nn_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "lightgreen", high = "green") +
  ggtitle("Neural Network Confusion Matrix") +
  theme_minimal()

# SVM
svm_cls <- svm(G3_class ~ . -G3, data = train_cls)
svm_cls_pred <- predict(svm_cls, test_cls)

# Confusion Matrix
svm_cls_pred <- factor(svm_cls_pred, levels = levels_to_use)
conf_svm <- confusionMatrix(svm_cls_pred, test_cls$G3_class)


# Confusion Matrix Plot
svm_cm <- table(Predicted = svm_cls_pred, Actual = test_cls$G3_class)
ggplot(as.data.frame(svm_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "pink", high = "red") +
  ggtitle("SVM Confusion Matrix") +
  theme_minimal()



# k-NN
train_cls_scaled <- scale(train_cls[, -which(names(train_cls) %in% c("G3", "G3_class"))])
test_cls_scaled <- scale(test_cls[, -which(names(test_cls) %in% c("G3", "G3_class"))],
                         center = attr(train_cls_scaled, "scaled:center"),
                         scale = attr(train_cls_scaled, "scaled:scale"))
knn_cls_pred <- knn(train = train_cls_scaled, test = test_cls_scaled,
                    cl = train_cls$G3_class, k = 5)

# Confusion Matrix
knn_cls_pred <- factor(knn_cls_pred, levels = levels_to_use)
conf_knn <- confusionMatrix(knn_cls_pred, test_cls$G3_class)


# Confusion Matrix Plot
knn_cm <- table(Predicted = knn_cls_pred, Actual = test_cls$G3_class)
ggplot(as.data.frame(knn_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "lightyellow", high = "gold") +
  ggtitle("k-NN Confusion Matrix") +
  theme_minimal()


# Match factor levels
rf_cls_pred <- factor(rf_cls_pred, levels = levels_to_use)
print(rf_cls_pred)
nn_cls_pred <- factor(nn_cls_pred, levels = levels_to_use)
print(nn_cls_pred)
svm_cls_pred <- factor(svm_cls_pred, levels = levels_to_use)
print(svm_cls_pred)
knn_cls_pred <- factor(knn_cls_pred, levels = levels_to_use)
print(knn_cls_pred)
test_cls$G3_class <- factor(test_cls$G3_class, levels = levels_to_use)
print(test_cls$G3_class )

# Evaluation
conf_rf <- confusionMatrix(rf_cls_pred, test_cls$G3_class)
print(conf_rf)
conf_nn <- confusionMatrix(nn_cls_pred, test_cls$G3_class)
print(conf_nn)
conf_svm <- confusionMatrix(svm_cls_pred, test_cls$G3_class)
print(conf_svm)
conf_knn <- confusionMatrix(knn_cls_pred, test_cls$G3_class)
print(conf_knn)

# Accuracy and AUC table
results <- data.frame(
  Model = c("Random Forest", "Neural Network", "SVM", "k-NN"),
  Accuracy = c(
    conf_rf$overall["Accuracy"],
    conf_nn$overall["Accuracy"],
    conf_svm$overall["Accuracy"],
    conf_knn$overall["Accuracy"]
  ),
  AUC = c(
    suppressMessages(multiclass.roc(test_cls$G3_class, as.numeric(rf_cls_pred))$auc),
    suppressMessages(multiclass.roc(test_cls$G3_class, as.numeric(nn_cls_pred))$auc),
    suppressMessages(multiclass.roc(test_cls$G3_class, as.numeric(svm_cls_pred))$auc),
    suppressMessages(multiclass.roc(test_cls$G3_class, as.numeric(knn_cls_pred))$auc)
  )
)


cat("Model Performance Summary:\n");print(results)


# Accuracy Comparison Bar Plot
ggplot(results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  ylim(0, 1) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy", x = "Model") +
  theme_minimal() +
  theme(legend.position = "none")

# AUC Comparison Bar Plot
ggplot(results, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity") +
  ylim(0, 1) +
  labs(title = "Model AUC Comparison", y = "AUC", x = "Model") +
  theme_minimal() +
  theme(legend.position = "none")


# Accuracy Summary
classification_summary <- data.frame(
  Model = c("Random Forest", "Neural Network", "SVM", "k-NN"),
  Accuracy = c(conf_rf$overall["Accuracy"], conf_nn$overall["Accuracy"],
               conf_svm$overall["Accuracy"], conf_knn$overall["Accuracy"])
)


#________________
# Cross Validation
#_________________

# 10-fold cross-validation setup
ctrl <- trainControl(method = "cv", number = 10)

# Train Random Forest with cross-validation on the regression task
rf_cv <- train(G3 ~ ., data = data_num, method = "rf", trControl = ctrl)

# Print results
print("Random Forest with 10-Fold Cross-Validation (Regression):");print(rf_cv)


# 10-fold cross-validation for classification
rf_cv_cls <- train(G3_class ~ . -G3, data = data_num, method = "rf", trControl = ctrl)

# Print classification CV results
print("Random Forest with 10-Fold Cross-Validation (Classification):");print(rf_cv_cls)
print(rf_cv_cls)

# remove all variables from the environment
detach()

rm(list=ls())

