# Bank Term Deposit Prediction using Machine Learning

## a. Problem Statement

The objective of this project is to predict whether a bank customer will subscribe to a term deposit based on demographic information and details related to the bank's marketing campaign. Five classification algorithms are implemented and compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive Streamlit application is provided to evaluate the models on test data.

## b. Dataset Description

Dataset: Bank Marketing / Bank Term Deposit Classification

The supplied CSV contains 11,162 instances and 17 columns. There are 16 input features and one binary target column, `deposit`.

Target classes:
- `no`: 5,873 records (52.62%)
- `yes`: 5,289 records (47.38%)

The dataset contains 7 numerical features and 9 categorical features. There are no missing values and no duplicate rows in the supplied CSV.

### Preprocessing

1. The target `deposit` is encoded as `no = 0` and `yes = 1`.
2. The data is divided into training and testing sets using an 80:20 stratified split with `random_state=42`.
3. Numerical features are standardized using `StandardScaler`.
4. Categorical features are converted using `OneHotEncoder(handle_unknown="ignore")`.
5. The preprocessing transformer is fitted only on the training data and then applied to the test data to avoid data leakage.

The resulting processed feature matrix contains 51 columns after one-hot encoding.

## c. Github Repository Link

https://github.com/Laksh910-hub/bank_term_deposit__ml

## Live Streamlit App Link

https://bank-term-deposit-prediction-2025ac05124.streamlit.app/

## d. Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8262 | 0.9071 | 0.8278 | 0.7996 | 0.8135 | 0.6513 |
| Decision Tree | 0.8110 | 0.8798 | 0.8093 | 0.7864 | 0.7977 | 0.6207 |
| KNN | 0.8173 | 0.8796 | 0.8199 | 0.7873 | 0.8033 | 0.6333 |
| Naive Bayes | 0.7201 | 0.8042 | 0.7837 | 0.5652 | 0.6568 | 0.4472 |
| Random Forest (Ensemble) | **0.8513** | **0.9173** | 0.8190 | **0.8809** | **0.8488** | **0.7047** |

### Observations on Model Performance

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression provided a strong baseline, with 82.62% accuracy and 0.9071 AUC. It achieved the highest precision (0.8278) among the five models, although its recall was lower than Random Forest. |
| Decision Tree | Decision Tree achieved 81.10% accuracy and 0.8798 AUC. Its performance was below Logistic Regression, KNN, and Random Forest on the overall set of metrics. |
| KNN | KNN achieved 81.73% accuracy and an F1 score of 0.8033. It performed competitively, but its AUC and recall were below those of the stronger models. |
| Naive Bayes | Naive Bayes had the lowest overall performance, with 72.01% accuracy, 0.8042 AUC, 0.6568 F1, and 0.4472 MCC. Its recall of 0.5652 was particularly low. |
| Random Forest (Ensemble) | Random Forest produced the strongest overall performance, with 85.13% accuracy, 0.9173 AUC, 0.8809 recall, 0.8488 F1, and 0.7047 MCC. Its precision was slightly below Logistic Regression. |

### Overall Winner

•	**Random Forest** is the overall best-performing model for the Bank Term Deposit dataset. 
•	It achieved the highest **Accuracy (85.13%), AUC (91.73%), Recall (88.09%), F1 Score (84.88%), and MCC (70.47%)** among the five models. 
•	Its ensemble approach allows it to capture more complex relationships in the customer data, resulting in better overall classification performance.

## Streamlit Application Features

The application provides:

•	Upload the held-out test CSV dataset
•	Select a classification model dropdown
•	View uploaded test data
•	Display below Evaluation metrics for the selected model
•	Display Accuracy 
•	Display AUC 
•	Display Precision 
•	Display Recall 
•	Display F1 Score 
•	Display MCC 
•	Display Confusion Matrix 
•	Display Classification Report 
•	Comparison of all five trained models 
•	Prediction Summary
•	View uploaded test data

