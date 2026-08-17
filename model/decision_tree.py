from sklearn.tree import DecisionTreeClassifier


def build_model():
    return DecisionTreeClassifier(max_depth=6, random_state=42)
