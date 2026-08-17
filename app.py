import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.logistic_regression import build_model as build_logistic
from model.decision_tree import build_model as build_tree
from model.knn import build_model as build_knn
from model.naive_bayes import build_model as build_naive_bayes
from model.random_forest import build_model as build_random_forest


st.set_page_config(
    page_title="Bank Deposit ML Lab",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERICAL_FEATURES = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome"
]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "deposit"

MODEL_BUILDERS = {
    "Logistic Regression": build_logistic,
    "Decision Tree": build_tree,
    "KNN": build_knn,
    "Naive Bayes": build_naive_bayes,
    "Random Forest": build_random_forest,
}

MODEL_NOTES = {
    "Logistic Regression": "A linear classifier that provides a strong baseline for binary prediction.",
    "Decision Tree": "A rule-based tree model that captures nonlinear feature relationships.",
    "KNN": "A distance-based classifier that predicts from nearby training observations.",
    "Naive Bayes": "A probabilistic classifier based on the Gaussian Naive Bayes assumption.",
    "Random Forest": "An ensemble of decision trees designed to improve generalization and robustness.",
}

EXPECTED_RESULTS = pd.DataFrame([
    ["Logistic Regression", 0.8262, 0.9071, 0.8278, 0.7996, 0.8135, 0.6513],
    ["Decision Tree", 0.8110, 0.8798, 0.8093, 0.7864, 0.7977, 0.6207],
    ["KNN", 0.8173, 0.8796, 0.8199, 0.7873, 0.8033, 0.6333],
    ["Naive Bayes", 0.7201, 0.8042, 0.7837, 0.5652, 0.6568, 0.4472],
    ["Random Forest", 0.8513, 0.9173, 0.8190, 0.8809, 0.8488, 0.7047],
], columns=["ML Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])


@st.cache_data
def load_training_data():
    return pd.read_csv("bank.csv")


@st.cache_resource
def prepare_and_train():
    df = load_training_data()
    X = df.drop(TARGET, axis=1)
    y = df[TARGET].map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    trained_models = {}
    for name, builder in MODEL_BUILDERS.items():
        model = builder()
        if name == "Naive Bayes":
            train_input = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
        else:
            train_input = X_train_processed
        model.fit(train_input, y_train)
        trained_models[name] = model

    return df, preprocessor, trained_models, X_test, y_test, X_test_processed


def to_prediction_matrix(matrix, model_name):
    if model_name == "Naive Bayes" and hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def evaluate_model(model, model_name, X_data, y_data):
    prediction_input = to_prediction_matrix(X_data, model_name)
    predictions = model.predict(prediction_input)
    probabilities = model.predict_proba(prediction_input)[:, 1]

    return {
        "Accuracy": accuracy_score(y_data, predictions),
        "AUC": roc_auc_score(y_data, probabilities),
        "Precision": precision_score(y_data, predictions, zero_division=0),
        "Recall": recall_score(y_data, predictions, zero_division=0),
        "F1": f1_score(y_data, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_data, predictions),
        "predictions": predictions,
    }


# Fix indentation of cached functions when this file is copied into environments that preserve decorators.
df, preprocessor, models, X_reference_test, y_reference_test, _ = prepare_and_train()

st.title("🏦 Bank Term Deposit ML Lab")
st.caption("Interactive comparison of five classification models on the Bank Marketing dataset")

st.markdown(
    """
    **Objective:** Predict whether a customer will subscribe to a term deposit (`yes` / `no`).  
    The application uses the same preprocessing, train/test split, and model settings used for the assignment experiments.
    """
)

with st.sidebar:
    st.header("Evaluation Controls")
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    selected_model_name = st.selectbox("Choose a classification model", list(MODEL_BUILDERS.keys()))
    st.divider()
    st.write("**Models available**")
    for model_name in MODEL_BUILDERS:
        st.write(f"• {model_name}")

st.subheader("Dataset Overview")
info1, info2, info3, info4 = st.columns(4)
info1.metric("Instances", f"{len(df):,}")
info2.metric("Input Features", len(FEATURES))
info3.metric("Target Classes", 2)
info4.metric("Missing Values", int(df.isna().sum().sum()))

if uploaded_file is None:
    st.info("Upload `test_data.csv` from the sidebar to evaluate the selected model.")

    st.subheader("Target Distribution")
    target_counts = df[TARGET].value_counts().rename_axis("deposit").reset_index(name="count")
    st.dataframe(target_counts, hide_index=True, use_container_width=True)

    st.subheader("Model Guide")
    guide = pd.DataFrame({"Model": list(MODEL_NOTES.keys()), "Purpose": list(MODEL_NOTES.values())})
    st.dataframe(guide, hide_index=True, use_container_width=True)
else:
    test_data = pd.read_csv(uploaded_file)

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in test_data.columns]

    if missing_columns:
        st.error("The uploaded CSV is missing required columns.")
        st.write(missing_columns)
        st.stop()

    if test_data[TARGET].isna().any():
        st.error("The uploaded target column contains missing values.")
        st.stop()

    if not test_data[TARGET].isin(["yes", "no"]).all():
        st.error("The `deposit` column must contain only `yes` and `no`.")
        st.stop()

    X_uploaded = test_data[FEATURES]
    y_uploaded = test_data[TARGET].map({"no": 0, "yes": 1})
    X_uploaded_processed = preprocessor.transform(X_uploaded)

    st.subheader("Uploaded Test Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Rows", f"{len(test_data):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Duplicate Rows", int(test_data.duplicated().sum()))
    st.dataframe(test_data.head(10), hide_index=True, use_container_width=True)

    selected_model = models[selected_model_name]
    result = evaluate_model(selected_model, selected_model_name, X_uploaded_processed, y_uploaded)

    st.divider()
    st.subheader(f"📊 {selected_model_name} Performance")
    st.caption(MODEL_NOTES[selected_model_name])

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{result['Accuracy']:.4f}")
    m2.metric("AUC", f"{result['AUC']:.4f}")
    m3.metric("Precision", f"{result['Precision']:.4f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Recall", f"{result['Recall']:.4f}")
    m5.metric("F1 Score", f"{result['F1']:.4f}")
    m6.metric("MCC", f"{result['MCC']:.4f}")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_uploaded, result["predictions"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm)
        ax.set_title(f"{selected_model_name} - Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_yticks([0, 1], ["No", "Yes"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Classification Report")
        report = classification_report(
            y_uploaded,
            result["predictions"],
            target_names=["No Deposit", "Deposit"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

    st.divider()
    st.subheader("All-Model Comparison on Uploaded Test Data")

    comparison_rows = []
    for model_name, model in models.items():
        model_result = evaluate_model(model, model_name, X_uploaded_processed, y_uploaded)
        comparison_rows.append([
            model_name,
            model_result["Accuracy"],
            model_result["AUC"],
            model_result["Precision"],
            model_result["Recall"],
            model_result["F1"],
            model_result["MCC"],
        ])

    comparison_df = pd.DataFrame(
        comparison_rows,
        columns=["ML Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    )
    st.dataframe(comparison_df.round(4), hide_index=True, use_container_width=True)

    winner = comparison_df.loc[comparison_df["F1"].idxmax(), "ML Model"]
    st.success(f"Overall winner on the uploaded test data by F1 Score: **{winner}**")
