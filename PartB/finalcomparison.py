# ============================================================
# PART C: FINAL COMPARISON
# Part A vs Part B (Class Weighting) vs Part B (Random Oversampling)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
)

# ------------------------------------------------------------
# 1. Put all saved probability outputs here
# ------------------------------------------------------------
model_results = {
    "Part A - Logistic Regression": lr_prob_A,
    "Part A - Random Forest": rf_prob_A,
    "Part A - Neural Network": nn_prob_A,

    "Part B (Class Weighting) - Logistic Regression": lr_prob_CW,
    "Part B (Class Weighting) - Random Forest": rf_prob_CW,
    "Part B (Class Weighting) - Neural Network": nn_prob_CW,

    "Part B (Oversampling) - Logistic Regression": lr_prob_ROS,
    "Part B (Oversampling) - Random Forest": rf_prob_ROS,
    "Part B (Oversampling) - Neural Network": nn_prob_ROS
}

comparison_data = []
pr_plot_data = {}
roc_plot_data = {}

# ------------------------------------------------------------
# 2. Calculate metrics
# ------------------------------------------------------------
for name, y_prob in model_results.items():
    y_pred = (y_prob >= 0.5).astype(int)

    acc    = accuracy_score(y_test, y_pred)
    prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_0  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_1  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wt  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    pr_plot_data[name]  = (recall, precision, pr_auc)
    roc_plot_data[name] = (fpr, tpr, roc_auc)

    cm = confusion_matrix(y_test, y_pred)
    cm_str = f"TN:{cm[0,0]} FP:{cm[0,1]} | FN:{cm[1,0]} TP:{cm[1,1]}"

    comparison_data.append({
        "Model":                name,
        "Accuracy":             round(acc,    4),
        "Precision (Class 0)":  round(prec_0, 4),
        "Precision (Class 1)":  round(prec_1, 4),
        "Recall (Class 0)":     round(rec_0,  4),
        "Recall (Class 1)":     round(rec_1,  4),
        "F1 Macro":             round(f1_mac, 4),
        "F1 Weighted":          round(f1_wt,  4),
        "ROC-AUC":              round(roc_auc, 4),
        "PR-AUC":               round(pr_auc,  4),
        "Confusion Matrix":     cm_str
    })

# ------------------------------------------------------------
# 3. Comparison table
# ------------------------------------------------------------
df_comparison = pd.DataFrame(comparison_data)

print("\n" + "=" * 100)
print("FINAL COMPARISON TABLE")
print("=" * 100)
display(df_comparison)

# ------------------------------------------------------------
# 4. Precision-Recall Curve Comparison
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))

for name, (recall, precision, pr_auc) in pr_plot_data.items():
    linestyle = "--" if "Part A" in name else "-"
    plt.plot(recall, precision, linestyle=linestyle, label=f"{name} (AUC = {pr_auc:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend(loc="lower left", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. ROC Curve Comparison
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))

for name, (fpr, tpr, roc_auc) in roc_plot_data.items():
    linestyle = "--" if "Part A" in name else "-"
    plt.plot(fpr, tpr, linestyle=linestyle, label=f"{name} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
