"""
Model Training & Evaluation Pipeline for Cardiovascular Risk & Multi-Vessel Stenosis.
Trains and evaluates models for:
1. Overall CAD Status (Cath)
2. LAD Stenosis
3. LCX Stenosis
4. RCA Stenosis
Strictly enforces Target Leakage Prevention and saves full classification metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS = ["Cath", "LAD", "LCX", "RCA"]
TARGET_DISPLAY_NAMES = {
    "Cath": "Overall CAD Status",
    "LAD": "Left Anterior Descending (LAD) Stenosis",
    "LCX": "Left Circumflex (LCX) Stenosis",
    "RCA": "Right Coronary Artery (RCA) Stenosis"
}

def get_candidate_models():
    """Return dictionary of candidate classifiers with tuned hyperparameters."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=3,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            C=0.5,
            max_iter=1000,
            penalty="l2",
            random_state=42
        )
    }

def train_and_evaluate(data_dir="data/processed", models_dir="models", docs_dir="docs"):
    """Train models for all 4 targets, evaluate cross-validation and holdout metrics."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    feature_names = list(X_train.columns)
    results = {}
    best_models = {}
    roc_data = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for target in TARGETS:
        print(f"\n==========================================")
        print(f" Training Models for Target: {target} ({TARGET_DISPLAY_NAMES[target]})")
        print(f"==========================================")

        y_tr = y_train[target]
        y_te = y_test[target]

        target_results = {"cv_benchmarks": {}, "test_evaluation": {}}
        best_model_name = None
        best_cv_auc = -1.0
        best_fitted_model = None

        candidate_models = get_candidate_models()

        for model_name, model in candidate_models.items():
            # 5-Fold Stratified Cross Validation
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            cv_scores = cross_validate(model, X_train, y_tr, cv=cv, scoring=scoring)

            mean_auc = float(np.mean(cv_scores["test_roc_auc"]))
            target_results["cv_benchmarks"][model_name] = {
                "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
                "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
                "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
                "cv_precision_std": float(np.std(cv_scores["test_precision"])),
                "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
                "cv_recall_std": float(np.std(cv_scores["test_recall"])),
                "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
                "cv_f1_std": float(np.std(cv_scores["test_f1"])),
                "cv_roc_auc_mean": mean_auc,
                "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"]))
            }

            print(f" -> {model_name:18s} | CV AUC: {mean_auc:.4f} | CV F1: {np.mean(cv_scores['test_f1']):.4f} | CV Acc: {np.mean(cv_scores['test_accuracy']):.4f}")

            if mean_auc > best_cv_auc:
                best_cv_auc = mean_auc
                best_model_name = model_name

        print(f"[*] Best model for {target}: {best_model_name} (CV ROC-AUC: {best_cv_auc:.4f})")

        # Fit best model on entire training set
        selected_model = candidate_models[best_model_name]
        selected_model.fit(X_train, y_tr)
        best_fitted_model = selected_model

        # Evaluate on Holdout Test Set
        y_pred = best_fitted_model.predict(X_test)
        if hasattr(best_fitted_model, "predict_proba"):
            y_prob = best_fitted_model.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_fitted_model.decision_function(X_test)

        test_acc = float(accuracy_score(y_te, y_pred))
        test_prec = float(precision_score(y_te, y_pred, zero_division=0))
        test_rec = float(recall_score(y_te, y_pred, zero_division=0))
        test_f1 = float(f1_score(y_te, y_pred, zero_division=0))
        test_auc = float(roc_auc_score(y_te, y_prob))
        cm = confusion_matrix(y_te, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_te, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_te, y_prob)
        pr_auc = float(auc(r_curve, p_curve))

        target_results["best_model"] = best_model_name
        target_results["test_evaluation"] = {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "roc_auc": round(test_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm,
            "test_sample_size": len(y_te)
        }

        roc_data[target] = {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "auc": round(test_auc, 4)
        }

        # Save model
        model_filename = f"{target.lower()}_model.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(best_fitted_model, model_path)
        best_models[target] = best_fitted_model

        results[target] = target_results
        print(f" Test Set Performance: Acc={test_acc:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

    # Save metrics JSON
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"metrics": results, "roc_curves": roc_data, "features": feature_names}, f, indent=2)
    print(f"\n[Training] Saved evaluation metrics to {metrics_path}")

    # Generate Evaluation Charts
    generate_evaluation_plots(roc_data, results, docs_dir)
    return results

def generate_evaluation_plots(roc_data, results, docs_dir):
    """Generate high-resolution ROC curves and benchmark summary plot."""
    plt.figure(figsize=(10, 8))
    colors = {"Cath": "#e63946", "LAD": "#f77f00", "LCX": "#457b9d", "RCA": "#2a9d8f"}
    labels = {
        "Cath": f"Overall CAD (AUC = {roc_data['Cath']['auc']:.3f})",
        "LAD": f"LAD Stenosis (AUC = {roc_data['LAD']['auc']:.3f})",
        "LCX": f"LCX Stenosis (AUC = {roc_data['LCX']['auc']:.3f})",
        "RCA": f"RCA Stenosis (AUC = {roc_data['RCA']['auc']:.3f})"
    }

    for target in TARGETS:
        plt.plot(roc_data[target]["fpr"], roc_data[target]["tpr"],
                 color=colors[target], lw=2.5, label=labels[target])

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance Level (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("Cardiovascular & Vessel Stenosis ROC-AUC Curves (Holdout Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_plot_path = os.path.join(docs_dir, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"[Training] Saved ROC plot to {roc_plot_path}")

if __name__ == "__main__":
    train_and_evaluate()
