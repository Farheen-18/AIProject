# Disaster Tweet Classification — Handling Class Imbalance

A machine learning project that classifies tweets as disaster-related or not, comparing the effect of different imbalance-handling techniques across three model types.

---

## Overview

This project uses the [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) dataset. Tweets are preprocessed, vectorized using TF-IDF, and classified using three models. The experiment is structured in three parts to study how class imbalance affects model performance and how different techniques address it.

---

## Project Structure

```
AIProjectFinal.ipynb   # Main notebook (all parts)
tweets.csv             # Input dataset (upload manually in Colab)
requirements.txt       # Python dependencies
README.md              # This file
```

---

## Experiment Design

### Part A — Baseline (No Imbalance Handling)
Trains three models on the raw imbalanced dataset.

### Part B — Imbalance Handling Techniques
Trains the same three models using two separate techniques:
- **Technique 1: Class Weighting** — Penalizes the model more for misclassifying the minority class.
- **Technique 2: Random Oversampling** — Duplicates minority class samples to balance the training set.

### Part C — Final Comparison
Aggregates all 9 model variants (3 models × 3 conditions) and compares them using a full metrics table, ROC curves, and Precision-Recall curves.

---

## Models Used

| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| Neural Network (MLP) | PyTorch |

---

## Evaluation Metrics

- Accuracy
- Precision & Recall (per class)
- F1 Score (Macro & Weighted)
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

## Setup & Usage

### 1. Clone or download the repository

```bash
git clone https://github.com/Farheen-18/AIProject.git
cd AIProject
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab (recommended)

Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Farheen-18/AIProject/blob/main/AIProjectFinal.ipynb)

When prompted, upload `tweets.csv` using the Colab file upload dialog.

### 4. Run locally (Jupyter)

```bash
jupyter notebook AIProjectFinal.ipynb
```

Make sure `tweets.csv` is in the same directory as the notebook. You may need to replace the `google.colab` file upload cell with a simple `pd.read_csv("tweets.csv")`.

---

## Requirements

```
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.66.0
seaborn>=0.13.0
joblib>=1.3.0
nltk>=3.8.0
regex>=2023.0.0
imbalanced-learn>=0.12.0

```

---

## Dataset

**File:** `tweets.csv`  
**Source:** [Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)  
**Columns used:** `text` (tweet content), `target` (0 = not disaster, 1 = disaster)

> The dataset is not included in this repository. Download it from Kaggle and place it in the project root before running.

---

## Text Preprocessing

Each tweet is cleaned by:
- Removing URLs and mentions (`@Farheen-18`)
- Removing hashtag symbols and non-alphabetic characters
- Lowercasing
- Stripping extra whitespace

Cleaned text is then vectorized using **TF-IDF** (max 5000 features, unigrams + bigrams).
