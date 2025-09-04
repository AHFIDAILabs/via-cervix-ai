import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, CLASS_NAMES

EVAL_LABELS = RESULTS_DIR / "eval_labels.npy"
EVAL_PROBS = RESULTS_DIR / "eval_probs.npy"

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, confidence_level=0.95):
    """Computes a metric and its confidence interval using bootstrapping."""
    metric_values = []
    for _ in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)))
        if len(np.unique(y_true[indices])) < 2: # Ensure both classes are present
            continue
        resampled_metric = metric_func(y_true[indices], y_pred[indices])
        metric_values.append(resampled_metric)
    
    lower_bound = np.percentile(metric_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(metric_values, (1 + confidence_level) / 2 * 100)
    mean_metric = np.mean(metric_values)
    return mean_metric, lower_bound, upper_bound

def sensitivity_specificity_ci(y_true, y_pred_probs, n_bootstrap=1000):
    """Calculates sensitivity and specificity with confidence intervals for each class."""
    n_classes = y_pred_probs.shape[1]
    results = {}

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (np.argmax(y_pred_probs, axis=1) == i).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n--- Class: {CLASS_NAMES[i]} ---")
        print(f"Sensitivity (Recall): {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
        
        # Bootstrap for AUC
        auc_ci = bootstrap_metric(y_true_binary, y_pred_probs[:, i], roc_auc_score)
        print(f"AUC: {auc_ci[0]:.3f} (95% CI: [{auc_ci[1]:.3f}, {auc_ci[2]:.3f}])")

def plot_confusion_matrix(cm, class_names, filepath):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(filepath)
    plt.close()
    print(f"\nConfusion matrix plot saved to {filepath}")

def main():
    """Main evaluation function."""
    if not EVAL_LABELS.exists() or not EVAL_PROBS.exists():
        print("Evaluation files not found. Please run training first.")
        return

    labels = np.load(EVAL_LABELS)
    probs = np.load(EVAL_PROBS)
    preds = np.argmax(probs, axis=1)

    print("="*50)
    print("           MODEL EVALUATION RESULTS")
    print("="*50)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    plot_confusion_matrix(cm, CLASS_NAMES, RESULTS_DIR / "confusion_matrix.png")

    print("\nClinical Metrics (with 95% Confidence Intervals):")
    sensitivity_specificity_ci(labels, probs)
    
    print("\n" + "="*50)


if __name__ == "__main__":
    main()