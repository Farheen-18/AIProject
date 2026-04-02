# ============================================================
# PART A: WITHOUT IMBALANCE HANDLING
# Models:
# 1. Logistic Regression
# 2. Random Forest
# 3. Neural Network
# ============================================================

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("tweets.csv")
df = df[["text", "target"]].dropna()

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["target"].value_counts())
print("\nClass distribution (%):")
print(df["target"].value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# 2. Clean text
# ------------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

# ------------------------------------------------------------
# 3. Split data
# ------------------------------------------------------------
X = df["clean_text"]
y = df["target"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 4. TF-IDF
# ------------------------------------------------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(X_train_text)
X_test  = tfidf.transform(X_test_text)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ------------------------------------------------------------
# 5. Evaluation function
# ------------------------------------------------------------
def evaluate_model(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Accuracy            :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision (class 0) :", round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4))
    print("Precision (class 1) :", round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4))
    print("Recall (class 0)    :", round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 4))
    print("Recall (class 1)    :", round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4))
    print("F1 Macro            :", round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4))
    print("F1 Weighted         :", round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4))
    print("ROC-AUC             :", round(roc_auc_score(y_true, y_prob), 4))

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    print("PR-AUC              :", round(pr_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Not Disaster", "Disaster"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PR Curve
    plt.figure(figsize=(5, 4))
    plt.plot(recall_curve, precision_curve, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 6. Logistic Regression
# ~~~~~~~~~~~~~~
# TUNABLE CONFIG
# ~~~~~~~~~~~~~~
LR_PENALTY   = "l2"
LR_C         = 1.0
LR_SOLVER    = "liblinear"
LR_THRESHOLD = 0.5
# ~~~~~~~~~~~~~~

print("\n--- Logistic Regression Config ---")
print(f"  Regularization : {LR_PENALTY}")
print(f"  C (strength)   : {LR_C}")
print(f"  Solver         : {LR_SOLVER}")
print(f"  Threshold      : {LR_THRESHOLD}")

lr_model = LogisticRegression(
    penalty=LR_PENALTY,
    C=LR_C,
    solver=LR_SOLVER,
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)
lr_prob = lr_model.predict_proba(X_test)[:, 1]
lr_prob_A = lr_prob.copy()
evaluate_model("Logistic Regression", y_test, lr_prob, threshold=LR_THRESHOLD)

# ------------------------------------------------------------
# 7. Random Forest
# ~~~~~~~~~~~~~~
# TUNABLE CONFIG
# ~~~~~~~~~~~~~~
RF_N_ESTIMATORS      = 100
RF_MAX_DEPTH         = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF  = 2
# ~~~~~~~~~~~~~~

print("\n--- Random Forest Config ---")
print(f"  N Estimators       : {RF_N_ESTIMATORS}")
print(f"  Max Depth          : {RF_MAX_DEPTH}")
print(f"  Min Samples Split  : {RF_MIN_SAMPLES_SPLIT}")
print(f"  Min Samples Leaf   : {RF_MIN_SAMPLES_LEAF}")

rf_model = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_split=RF_MIN_SAMPLES_SPLIT,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_prob_A = rf_prob.copy()
evaluate_model("Random Forest", y_test, rf_prob, threshold=0.5)

# ------------------------------------------------------------
# 8. Neural Network (PyTorch)
# Fixed architecture: Input -> 64 -> 32 -> 16 -> Output
# ~~~~~~~~~~~~~~
# TUNABLE CONFIG
# ~~~~~~~~~~~~~~
ACTIVATION    = "relu"
LEARNING_RATE = 0.001
OPTIMIZER     = "adam"
BATCH_SIZE    = 64
EPOCHS        = 20
DROPOUT       = 0.3
L2_REG        = 1e-4
INIT          = "xavier"
NN_THRESHOLD  = 0.5
# ~~~~~~~~~~~~~~

print("\n--- Neural Network Config ---")
print(f"  Activation     : {ACTIVATION}")
print(f"  Learning Rate  : {LEARNING_RATE}")
print(f"  Optimizer      : {OPTIMIZER}")
print(f"  Batch Size     : {BATCH_SIZE}")
print(f"  Epochs         : {EPOCHS}")
print(f"  Dropout        : {DROPOUT}")
print(f"  L2 Reg         : {L2_REG}")
print(f"  Weight Init    : {INIT}")
print(f"  Threshold      : {NN_THRESHOLD}")

class TextNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        act_map = {
            "relu":      nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "tanh":      nn.Tanh(),
            "elu":       nn.ELU()
        }
        act = act_map.get(ACTIVATION, nn.ReLU())

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

        self.act     = act
        self.dropout = nn.Dropout(DROPOUT)

        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            if INIT == "xavier":
                nn.init.xavier_uniform_(fc.weight)
            elif INIT == "he":
                if ACTIVATION in ["relu", "leakyrelu"]:
                    nn.init.kaiming_uniform_(fc.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(fc.weight)
            elif INIT == "uniform":
                nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.zeros_(fc.bias)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.act(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze(1)

# Prepare tensors
X_train_dense  = X_train.toarray().astype(np.float32)
X_test_dense   = X_test.toarray().astype(np.float32)
X_train_tensor = torch.tensor(X_train_dense)
y_train_tensor = torch.tensor(y_train.to_numpy().astype(np.float32))
X_test_tensor  = torch.tensor(X_test_dense)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

nn_model  = TextNeuralNet(input_dim=X_train_dense.shape[1])
criterion = nn.BCELoss()
opt_map   = {
    "adam":    torch.optim.Adam(nn_model.parameters(),    lr=LEARNING_RATE, weight_decay=L2_REG),
    "sgd":     torch.optim.SGD(nn_model.parameters(),     lr=LEARNING_RATE, weight_decay=L2_REG),
    "rmsprop": torch.optim.RMSprop(nn_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
}
optimizer = opt_map.get(OPTIMIZER, torch.optim.Adam(nn_model.parameters(), lr=LEARNING_RATE))

print("\nTraining Neural Network...")
for epoch in range(EPOCHS):
    nn_model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(nn_model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

nn_model.eval()
with torch.no_grad():
    nn_prob = nn_model(X_test_tensor).cpu().numpy()
    nn_prob_A = nn_prob.copy()

evaluate_model("Neural Network", y_test, nn_prob, threshold=NN_THRESHOLD)

print("\nPart A completed successfully.")
