import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "isolate_required"

# Project root = the folder that contains data/, models/, notebooks/ and src/.
# Works no matter which folder the script is started from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_router_telemetry_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"
CV_RESULTS_PATH = PROJECT_ROOT / "models" / "model_comparison_cv.csv"


def split_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_holdout, y_train, y_val, y_holdout


def save_splits(X_train, X_val, X_holdout, y_train, y_val, y_holdout):
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    val_df = X_val.copy()
    val_df[TARGET_COLUMN] = y_val.values

    holdout_df = X_holdout.copy()
    holdout_df[TARGET_COLUMN] = y_holdout.values

    train_df.to_csv(PROJECT_ROOT / "data" / "train_80.csv", index=False)
    val_df.to_csv(PROJECT_ROOT / "data" / "validation_10.csv", index=False)
    


def build_preprocessor(X):
    # "str" is included so this also works with pandas 3, which stores text as "str" dtype.
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def build_pipeline_and_param_grid(preprocessor):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression())
    ])

    param_grid = [
        {
            "model": [LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)],
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"]
        },
        {
            "model": [RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)],
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        },
        {
            "model": [SVC(probability=True, random_state=RANDOM_STATE)],
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        },
        {
            "model": [XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")],
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        },
        {
            "model": [KNeighborsClassifier()],
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["minkowski"]
        }
    ]

    return pipeline, param_grid


def compute_metrics_from_predictions(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def evaluate_model_default_threshold(model, X_val, y_val):
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = None

    return compute_metrics_from_predictions(y_val, y_pred, y_proba)


def tune_threshold(y_true, y_proba, min_precision=0.55):
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        precision = precision_score(y_true, y_pred_t, zero_division=0)
        recall = recall_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred_t)
        rows.append({
            "threshold": float(round(t, 2)),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        })

    threshold_df = pd.DataFrame(rows)

    best_f1_row = threshold_df.sort_values(
        by=["f1_score", "precision", "recall"],
        ascending=[False, False, False]
    ).iloc[0]

    precision_filtered = threshold_df[threshold_df["precision"] >= min_precision]

    if not precision_filtered.empty:
        best_precision_tradeoff_row = precision_filtered.sort_values(
            by=["f1_score", "recall", "threshold"],
            ascending=[False, False, True]
        ).iloc[0]
    else:
        best_precision_tradeoff_row = None

    return threshold_df, best_f1_row.to_dict(), None if best_precision_tradeoff_row is None else best_precision_tradeoff_row.to_dict()


def evaluate_model_with_threshold(model, X_val, y_val, threshold):
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics_from_predictions(y_val, y_pred, y_proba)
    metrics["threshold"] = float(threshold)
    return metrics


def get_model_name(model):
    return model.__class__.__name__


def train_model(df):
    df = df.copy()

    X_train, X_val, X_holdout, y_train, y_val, y_holdout = split_data(df, TARGET_COLUMN)
    save_splits(X_train, X_val, X_holdout, y_train, y_val, y_holdout)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    pipeline, param_grid = build_pipeline_and_param_grid(preprocessor)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    # Save every configuration tried (all 5 model families), not only the winner.
    cv_results = grid_search.cv_results_
    cv_summary = pd.DataFrame({
        "rank": cv_results["rank_test_score"],
        "model": [type(p["model"]).__name__ for p in cv_results["params"]],
        "mean_cv_f1": cv_results["mean_test_score"],
        "std_cv_f1": cv_results["std_test_score"],
        "params": [str({k: v for k, v in p.items() if k != "model"}) for p in cv_results["params"]],
    }).sort_values("rank")
    cv_summary.to_csv(CV_RESULTS_PATH, index=False)

    best_model = grid_search.best_estimator_
    best_model_name = get_model_name(best_model.named_steps["model"])

    default_val_metrics = evaluate_model_default_threshold(best_model, X_val, y_val)

    if not hasattr(best_model, "predict_proba"):
        raise ValueError(
            "Best model does not support predict_proba, threshold tuning cannot be performed."
        )

    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    threshold_df, best_f1_threshold_info, best_precision_tradeoff_info = tune_threshold(
        y_val, y_val_proba, min_precision=0.55
    )

    best_threshold = best_f1_threshold_info["threshold"]
    tuned_val_metrics = evaluate_model_with_threshold(best_model, X_val, y_val, best_threshold)

    holdout_default_metrics = evaluate_model_default_threshold(best_model, X_holdout, y_holdout)
    holdout_tuned_metrics = evaluate_model_with_threshold(best_model, X_holdout, y_holdout, best_threshold)

    print("\nBest model found:", best_model_name)
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation F1:", grid_search.best_score_)

    print("\nValidation metrics at default threshold (0.5):")
    print(default_val_metrics)

    print("\nBest validation threshold for F1:")
    print(best_f1_threshold_info)

    if best_precision_tradeoff_info is not None:
        print("\nBest validation threshold with precision >= 0.55:")
        print(best_precision_tradeoff_info)
    else:
        print("\nNo threshold achieved precision >= 0.55 on validation set.")

    print("\nValidation metrics at tuned threshold:")
    print(tuned_val_metrics)

    print("\nHoldout metrics at default threshold (0.5):")
    print(holdout_default_metrics)

    print("\nHoldout metrics at tuned threshold:")
    print(holdout_tuned_metrics)

    best_params = grid_search.best_params_.copy()
    best_params["model"] = best_model_name

    metadata = {
        # Relative path, so the metadata does not contain your personal C:\ path.
        "data_path": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "train_shape": X_train.shape,
        "validation_shape": X_val.shape,
        "holdout_shape": X_holdout.shape,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_cv_score": float(grid_search.best_score_),
        "validation_metrics_default_threshold": default_val_metrics,
        "best_validation_threshold_for_f1": best_f1_threshold_info,
        "best_validation_threshold_with_precision_constraint": best_precision_tradeoff_info,
        "validation_metrics_tuned_threshold": tuned_val_metrics,
        "holdout_metrics_default_threshold": holdout_default_metrics,
        "holdout_metrics_tuned_threshold": holdout_tuned_metrics,
        "selected_threshold_for_inference": float(best_threshold),
        "class_balance": {
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_val.mean()),
            "holdout_positive_rate": float(y_holdout.mean())
        }
    }

    return best_model, metadata


def main():
    df = pd.read_csv(DATA_PATH)

    best_model, metadata = train_model(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nBest model saved successfully.")
    print("Selected threshold for inference:", metadata["selected_threshold_for_inference"])


if __name__ == "__main__":
    main()