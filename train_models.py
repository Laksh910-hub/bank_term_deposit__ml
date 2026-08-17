import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("bank.csv")

print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

numerical_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

X = df.drop("deposit", axis=1)

y = df["deposit"].map({
    "no": 0,
    "yes": 1
})


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)

print(
    "Processed training shape:",
    X_train_processed.shape
)


# ============================================================
# 5. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("model", exist_ok=True)


# ============================================================
# 6. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "model/preprocessor.joblib"
)

print("Saved: model/preprocessor.joblib")


# ============================================================
# 7. TRAIN AND SAVE LOGISTIC REGRESSION
# ============================================================

logistic_regression = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_regression.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    logistic_regression,
    "model/logistic_regression.joblib"
)

print("Saved: model/logistic_regression.joblib")


# ============================================================
# 8. TRAIN AND SAVE DECISION TREE
# ============================================================

decision_tree = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

decision_tree.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    decision_tree,
    "model/decision_tree.joblib"
)

print("Saved: model/decision_tree.joblib")


# ============================================================
# 9. TRAIN AND SAVE KNN
# ============================================================

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    knn,
    "model/knn.joblib"
)

print("Saved: model/knn.joblib")


# ============================================================
# 10. TRAIN AND SAVE NAIVE BAYES
# ============================================================

naive_bayes = GaussianNB()

# GaussianNB requires a dense matrix
X_train_nb = (
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

naive_bayes.fit(
    X_train_nb,
    y_train
)

joblib.dump(
    naive_bayes,
    "model/naive_bayes.joblib"
)

print("Saved: model/naive_bayes.joblib")


# ============================================================
# 11. TRAIN AND SAVE RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    random_forest,
    "model/random_forest.joblib"
)

print("Saved: model/random_forest.joblib")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL SAVING COMPLETED")
print("=" * 60)

print("\nSaved files:")

for filename in [
    "preprocessor.joblib",
    "logistic_regression.joblib",
    "decision_tree.joblib",
    "knn.joblib",
    "naive_bayes.joblib",
    "random_forest.joblib"
]:
    print("✓ model/" + filename)

print("\nAll models trained using the same:")
print("• Dataset")
print("• 80/20 train-test split")
print("• random_state=42")
print("• Preprocessing")