library(caret)
library(e1071)
library(dplyr)
data <- read.csv("D:/Sem 10/Data Science/adult.csv",header = TRUE,sep = ",")
head(data)
col_names <- c('age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
               'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')
colnames(data) <- col_names
print(colnames(data))
head(data)
categorical <- names(data)[sapply(data, function(x) is.character(x) || is.factor(x))]
cat('There are', length(categorical), 'categorical variables\n')
cat('The categorical variables are:\n\n', categorical, '\n')
subset_data <- data[, categorical]
head(subset_data)
missing_counts <- colSums(is.na(data[, categorical]))
cat('Missing value counts in categorical columns:\n\n')
print(missing_counts)
for (var in categorical) {
  cat(paste("Variable:", var, "\n"))
  print(table(data[[var]]))
  cat("\n")
}
for (var in categorical) {
  cat(paste("Variable:", var, "\n"))
  print(table(data[[var]]) / nrow(data))
  cat("\n")
}
unique_values <- unique(data$workclass)
print(unique_values)
workclass_counts <- table(data$workclass)
print(workclass_counts)
data$workclass[data$workclass == '?'] <- NA
workclass_counts <- table(data$workclass)
print(workclass_counts)

unique_values <- unique(data$occupation)
print(unique_values)
occupation_counts <- table(data$occupation)
print(occupation_counts)
data$occupation[data$occupation == '?'] <- NA
occupation_counts <- table(data$occupation)
print(occupation_counts)

unique_values <- unique(data$native_country)
print(unique_values)
native_country_counts <- table(data$native_country)
print(native_country_counts)
data$native_country[data$native_country == '?'] <- NA
native_country_counts <- table(data$native_country)
print(native_country_counts)

missing_counts <- colSums(is.na(data[, categorical]))
cat('Missing value counts in categorical columns:\n\n')
print(missing_counts)

categorical_missing <- names(missing_counts[missing_counts > 0])
for (var in categorical_missing) {
  mode_val <- names(sort(table(data[[var]], exclude = NULL), decreasing = TRUE))[1]
  data[[var]][is.na(data[[var]])] <- mode_val
}

missing_counts_after_imputation <- colSums(is.na(data[, categorical]))
cat('Missing value counts in categorical columns after imputation:\n\n')
print(missing_counts_after_imputation)


for (var in categorical) {
  cat(paste(var, ' contains ', length(unique(data[[var]])), ' labels'), "\n")
}

numerical <- names(data)[sapply(data, function(x) !is.character(x))]
cat('There are', length(numerical), 'numerical variables\n')
cat('The numerical variables are:', numerical, '\n')
subset_data <- data[, numerical]
head(subset_data)

missing_counts <- colSums(is.na(data[, numerical]))
cat('Missing value counts in numerical columns:\n\n')
print(missing_counts)
for (var in categorical) {
  contingency_table <- table(data[[var]], data$income)
  chi_squared_result <- chisq.test(contingency_table)
  
  cat(paste("Variable:", var, "\n"))
  print(chi_squared_result)
  cat("\n")
}

data$income <- as.factor(data$income)
set.seed(123)
split_index <- createDataPartition(data$income, p = 0.7, list = FALSE)
train_data <- data[split_index, ]
test_data <- data[-split_index, ]

naive_bayes_model <- naiveBayes(income ~ ., data = train_data)
predictions <- predict(naive_bayes_model, newdata = test_data)
accuracy <- sum(predictions == test_data$income) / nrow(test_data)
cat('Accuracy on the test set:', accuracy, '\n')

set.seed(123)
train_control <- trainControl(method = 'cv', number = 10)
naive_bayes_model_cv <- train(
  income ~ .,
  data = data,
  method = 'naive_bayes',
  trControl = train_control
)

print(naive_bayes_model_cv)
accuracy_cv <- naive_bayes_model_cv$results$Accuracy
cat('Accuracy using 10-fold cross-validation:', accuracy_cv, '\n')

conf_matrix <- confusionMatrix(predictions, test_data$income)
cat('Confusion Matrix:\n')
print(conf_matrix$table)
recall <- conf_matrix$byClass['Sensitivity']
precision <- conf_matrix$byClass['Pos Pred Value']
f_measure <- 2 * (precision * recall) / (precision + recall)

cat('\nRecall (Sensitivity):', recall, '\n')
cat('Precision (Pos Pred Value):', precision, '\n')
cat('F-measure:', f_measure, '\n')

conf_matrix <- confusionMatrix(predictions, test_data$income)
cat('Confusion Matrix:\n\n')
print(conf_matrix$table)
TP <- conf_matrix$table[2, 2]
TN <- conf_matrix$table[1, 1]
FP <- conf_matrix$table[1, 2]
FN <- conf_matrix$table[2, 1]

cat('\nTrue Positives (TP):', TP, '\n')
cat('True Negatives (TN):', TN, '\n')
cat('False Positives (FP):', FP, '\n')
cat('False Negatives (FN):', FN, '\n')

