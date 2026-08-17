import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Bank Term Deposit Prediction")

st.markdown(
    """
    ### Machine Learning Classification Model Comparison

    This application predicts whether a bank customer will
    subscribe to a term deposit using trained classification
    models.
    """
)

st.divider()


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        "model/preprocessor.joblib"
    )

    models = {

        "Logistic Regression":
            joblib.load(
                "model/logistic_regression.joblib"
            ),

        "Decision Tree":
            joblib.load(
                "model/decision_tree.joblib"
            ),

        "KNN":
            joblib.load(
                "model/knn.joblib"
            ),

        "Naive Bayes":
            joblib.load(
                "model/naive_bayes.joblib"
            ),

        "Random Forest":
            joblib.load(
                "model/random_forest.joblib"
            )
    }

    return preprocessor, models


preprocessor, models = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data CSV",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)


# ============================================================
# MODEL INFORMATION
# ============================================================

model_descriptions = {

    "Logistic Regression":
        "Linear classification model that estimates the probability of a customer subscribing to a term deposit.",

    "Decision Tree":
        "Tree-based classifier that makes predictions through a sequence of feature-based decisions.",

    "KNN":
        "Instance-based classifier that predicts a class using the nearest training observations.",

    "Naive Bayes":
        "Probabilistic classifier based on Bayes' theorem with the assumption of conditional independence between features.",

    "Random Forest":
        "Ensemble classifier that combines multiple decision trees to improve predictive performance and robustness."
}


st.sidebar.info(
    model_descriptions[selected_model_name]
)


# ============================================================
# INITIAL SCREEN
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload the test_data.csv file using the "
        "sidebar to evaluate the selected model."
    )

    st.subheader("Available Classification Models")

    for model_name in models.keys():

        st.write(
            f"• {model_name}"
        )


# ============================================================
# PROCESS UPLOADED DATA
# ============================================================

else:

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    test_data = pd.read_csv(
        uploaded_file
    )


    # --------------------------------------------------------
    # DISPLAY DATA INFORMATION
    # --------------------------------------------------------

    st.subheader("Uploaded Test Data")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        test_data.shape[0]
    )

    col2.metric(
        "Columns",
        test_data.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(test_data.isnull().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(test_data.duplicated().sum())
    )

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


    # --------------------------------------------------------
    # VALIDATE TARGET COLUMN
    # --------------------------------------------------------

    if "deposit" not in test_data.columns:

        st.error(
            "The uploaded CSV must contain the 'deposit' target column."
        )

        st.stop()


    # --------------------------------------------------------
    # PREPARE FEATURES AND TARGET
    # --------------------------------------------------------

    X_uploaded = test_data.drop(
        "deposit",
        axis=1
    )

    y_uploaded = test_data["deposit"].map({
        "no": 0,
        "yes": 1
    })


    # Check for invalid target values

    if y_uploaded.isnull().any():

        st.error(
            "The 'deposit' column must contain only 'yes' or 'no'."
        )

        st.stop()


    # --------------------------------------------------------
    # TRANSFORM FEATURES
    # --------------------------------------------------------

    X_uploaded_processed = preprocessor.transform(
        X_uploaded
    )


    # --------------------------------------------------------
    # SELECT TRAINED MODEL
    # --------------------------------------------------------

    selected_model = models[
        selected_model_name
    ]


    # --------------------------------------------------------
    # PREPARE INPUT FOR MODEL
    # --------------------------------------------------------

    if selected_model_name == "Naive Bayes":

        X_prediction = (
            X_uploaded_processed.toarray()
            if hasattr(
                X_uploaded_processed,
                "toarray"
            )
            else X_uploaded_processed
        )

    else:

        X_prediction = X_uploaded_processed


    # --------------------------------------------------------
    # MAKE PREDICTIONS
    # --------------------------------------------------------

    y_pred = selected_model.predict(
        X_prediction
    )

    y_prob = selected_model.predict_proba(
        X_prediction
    )[:, 1]


    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_uploaded,
        y_pred
    )

    auc = roc_auc_score(
        y_uploaded,
        y_prob
    )

    precision = precision_score(
        y_uploaded,
        y_pred
    )

    recall = recall_score(
        y_uploaded,
        y_pred
    )

    f1 = f1_score(
        y_uploaded,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_uploaded,
        y_pred
    )


    # ========================================================
    # EVALUATION METRICS
    # ========================================================

    st.divider()

    st.subheader(
        f"{selected_model_name} Evaluation Metrics"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )


    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.4f}"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_uploaded,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.imshow(cm)

    ax.set_title(
        f"{selected_model_name} - Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["No", "Yes"]
    )

    ax.set_yticklabels(
        ["No", "Yes"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.divider()

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_uploaded,
        y_pred,
        target_names=[
            "No Deposit",
            "Deposit"
        ],
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    # ========================================================
    # ALL MODEL COMPARISON TABLE
    # ========================================================

    st.divider()

    st.subheader(
        "Model Comparison"
    )

    st.markdown(
        """
        The table below compares all five trained classification
        models using the six required evaluation metrics.
        """
    )

    comparison_results = []

    for model_name, model in models.items():

        # Naive Bayes requires dense input
        if model_name == "Naive Bayes":

            X_model_prediction = (
                X_uploaded_processed.toarray()
                if hasattr(
                    X_uploaded_processed,
                    "toarray"
                )
                else X_uploaded_processed
            )

        else:

            X_model_prediction = X_uploaded_processed


        # Predictions
        model_pred = model.predict(
            X_model_prediction
        )

        model_prob = model.predict_proba(
            X_model_prediction
        )[:, 1]


        # Metrics
        model_accuracy = accuracy_score(
            y_uploaded,
            model_pred
        )

        model_auc = roc_auc_score(
            y_uploaded,
            model_prob
        )

        model_precision = precision_score(
            y_uploaded,
            model_pred
        )

        model_recall = recall_score(
            y_uploaded,
            model_pred
        )

        model_f1 = f1_score(
            y_uploaded,
            model_pred
        )

        model_mcc = matthews_corrcoef(
            y_uploaded,
            model_pred
        )


        comparison_results.append({

            "ML Model":
                model_name,

            "Accuracy":
                model_accuracy,

            "AUC":
                model_auc,

            "Precision":
                model_precision,

            "Recall":
                model_recall,

            "F1":
                model_f1,

            "MCC":
                model_mcc
        })


    comparison_df = pd.DataFrame(
        comparison_results
    )


    # Round values for display
    comparison_display = comparison_df.copy()

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    comparison_display[
        metric_columns
    ] = comparison_display[
        metric_columns
    ].round(4)


    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model_row = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    st.success(
        f"Best model based on Accuracy: "
        f"**{best_model_row['ML Model']}** "
        f"with an Accuracy of "
        f"**{best_model_row['Accuracy']:.4f}**."
    )

    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Summary"
    )

    prediction_counts = pd.Series(
        y_pred
    ).value_counts()

    summary_col1, summary_col2 = st.columns(2)

    summary_col1.metric(
        "Predicted No Deposit",
        int(prediction_counts.get(0, 0))
    )

    summary_col2.metric(
        "Predicted Deposit",
        int(prediction_counts.get(1, 0))
    )