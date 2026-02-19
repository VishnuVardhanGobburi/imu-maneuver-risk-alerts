"""
Maneuver-type prediction: train a classifier on window features, report metrics and importance.
Uses Random Forest (so all 4 classes get predicted; LR was collapsing to 1 and 4).
"""

import pandas as pd
import numpy as np
from typing import Any

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def get_feature_matrix_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (all numeric columns except Target) and y (Target). Drops rows with NaN in Target."""
    if "Target" not in df.columns:
        raise ValueError("DataFrame must contain 'Target' column")
    out = df.dropna(subset=["Target"]).copy()
    y = out["Target"].astype(int)
    X = out.drop(columns=["Target"])
    # Keep only numeric feature columns
    X = X.select_dtypes(include=[np.number])
    return X, y


def _all_classes_present(y_train: pd.Series, y_test: pd.Series, class_names: list) -> bool:
    """Return True iff both train and test contain every class."""
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    required = set(class_names)
    return (required <= train_classes) and (required <= test_classes)


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
    use_ordered_split: bool = False,
) -> dict[str, Any]:
    """
    Build feature matrix, impute, scale, split (stratified so all classes in train and test), train Logistic Regression, evaluate.
    If use_ordered_split: train on first (1-test_size), test on last test_size by row order; fails if any class is missing in either split.
    Otherwise: stratified random split (guarantees all classes in both splits when possible).
    Returns dict with: model, accuracy, report_dict, confusion_matrix, feature_importances, class_names, n_train, n_test.
    """
    if not SKLEARN_AVAILABLE:
        return {
            "error": "scikit-learn is required. Install with: pip install scikit-learn",
            "model": None,
            "scaler": None,
            "imputer": None,
            "feature_columns": None,
            "accuracy": None,
            "report_dict": None,
            "confusion_matrix": None,
            "feature_importances": None,
            "class_names": None,
            "n_train": None,
            "n_test": None,
            "use_ordered_split": False,
        }
    X, y = get_feature_matrix_and_target(df)
    if X.empty or len(y) < 10:
        return {
            "error": "Not enough data after dropping missing Target.",
            "model": None,
            "scaler": None,
            "imputer": None,
            "feature_columns": None,
            "accuracy": None,
            "report_dict": None,
            "confusion_matrix": None,
            "feature_importances": None,
            "class_names": None,
            "n_train": None,
            "n_test": None,
            "use_ordered_split": False,
        }
    class_names = sorted(y.unique().tolist())
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

    if use_ordered_split:
        n = len(X_imp)
        n_test = int(n * test_size)
        n_train = n - n_test
        if n_train < 5 or n_test < 5:
            return {
                "error": "Ordered split: not enough rows for train/test.",
                "model": None,
                "scaler": None,
                "imputer": None,
                "feature_columns": None,
                "accuracy": None,
                "report_dict": None,
                "confusion_matrix": None,
                "feature_importances": None,
                "class_names": class_names,
                "n_train": None,
                "n_test": None,
                "use_ordered_split": True,
            }
        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train, n)
        X_train = X_imp.iloc[train_idx]
        X_test = X_imp.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if not _all_classes_present(y_train, y_test, class_names):
            missing_train = set(class_names) - set(y_train.unique())
            missing_test = set(class_names) - set(y_test.unique())
            msg = "Ordered split: not all classes appear in both train and test. Use stratified (random) split."
            if missing_test:
                msg += f" Missing in test: {sorted(missing_test)}."
            if missing_train:
                msg += f" Missing in train: {sorted(missing_train)}."
            return {
                "error": msg,
                "model": None,
                "scaler": None,
                "imputer": None,
                "feature_columns": None,
                "accuracy": None,
                "report_dict": None,
                "confusion_matrix": None,
                "feature_importances": None,
                "class_names": class_names,
                "n_train": None,
                "n_test": None,
                "use_ordered_split": True,
            }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imp, y, test_size=test_size, random_state=random_state, stratify=y
        )
        if not _all_classes_present(y_train, y_test, class_names):
            return {
                "error": "Stratified split could not include all classes in both sets (e.g. a class has too few samples). Try a smaller test fraction or check the data.",
                "model": None,
                "scaler": None,
                "imputer": None,
                "feature_columns": None,
                "accuracy": None,
                "report_dict": None,
                "confusion_matrix": None,
                "feature_importances": None,
                "class_names": class_names,
                "n_train": None,
                "n_test": None,
                "use_ordered_split": False,
            }

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest: predicts all 4 classes (LR was collapsing to 1 and 4)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    # Feature importance from RF (impurity decrease)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {
        "error": None,
        "model": model,
        "scaler": scaler,
        "imputer": imp,
        "feature_columns": X.columns.tolist(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "class_names": class_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "use_ordered_split": use_ordered_split,
    }


def predict_one_row(
    row_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    imputer: Any,
    feature_columns: list[str],
) -> tuple[int | None, np.ndarray | None]:
    """
    Run the classification pipeline (impute, scale, predict) on a single row.
    row_df: 1-row DataFrame with at least the feature_columns (Target optional).
    Returns (predicted_class, proba_array) or (None, None) on error.
    """
    if not SKLEARN_AVAILABLE or model is None or scaler is None or imputer is None:
        return None, None
    try:
        X = row_df[feature_columns].copy() if all(c in row_df.columns for c in feature_columns) else row_df.reindex(columns=feature_columns)
        X_imp = imputer.transform(X)
        X_s = scaler.transform(X_imp)
        pred = int(model.predict(X_s)[0])
        proba = model.predict_proba(X_s)[0] if hasattr(model, "predict_proba") else None
        return pred, (proba if proba is not None else np.array([]))
    except Exception:
        return None, None
