import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop("deposit", axis=1)

y_test = test_data["deposit"].map({
    "no": 0,
    "yes": 1
})


# ============================================================
# 2. LOAD PREPROCESSOR
# ============================================================

preprocessor = joblib.load(
    "model/preprocessor.joblib"
)


# ============================================================
# 3. TRANSFORM TEST DATA
# ============================================================

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "Processed test shape:",
    X_test_processed.shape
)


# ============================================================
# 4. MODEL FILES
# ============================================================

model_files = {
    "Logistic Regression":
        "model/logistic_regression.joblib",

    "Decision Tree":
        "model/decision_tree.joblib",

    "KNN":
        "model/knn.joblib",

    "Naive Bayes":
        "model/naive_bayes.joblib",

    "Random Forest":
        "model/random_forest.joblib"
}


# ============================================================
# 5. VERIFY EACH SAVED MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVED MODEL VERIFICATION")
print("=" * 70)


for model_name, model_path in model_files.items():

    model = joblib.load(model_path)

    # Naive Bayes needs dense input
    if model_name == "Naive Bayes":

        X_prediction = (
            X_test_processed.toarray()
            if hasattr(X_test_processed, "toarray")
            else X_test_processed
        )

    else:

        X_prediction = X_test_processed


    # Predictions
    y_pred = model.predict(
        X_prediction
    )

    y_prob = model.predict_proba(
        X_prediction
    )[:, 1]


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )


    print("\n" + "-" * 70)

    print(model_name)

    print("-" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")