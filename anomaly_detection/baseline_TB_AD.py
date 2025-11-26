import os
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# =========================
# Config
# =========================
CSV_PATH = "../master_tables/TB/test"
SEED = 42

# =========================
# Data loading
# =========================
def load_csvs(csv_path):
    traces_df = pd.read_csv(os.path.join(csv_path, "traces.csv"))
    events_df = pd.read_csv(os.path.join(csv_path, "events.csv"))
    return traces_df, events_df

traces_df, events_df = load_csvs(CSV_PATH)

print("Building per-trace OpName + Description sequences...")

events_df["TaskID"] = events_df["TaskID"].astype(str)
traces_df["TaskID"] = traces_df["TaskID"].astype(str)

def normalize_desc(desc):
    if pd.isna(desc):
        return ""
    x = str(desc).lower()
    x = re.sub(r'0x[0-9a-f]+', ' ', x)     # remove hex ids
    x = re.sub(r'\b\d+\b', ' ', x)         # remove pure numbers
    x = re.sub(r'\s+', ' ', x)             # collapse spaces
    return x.strip()

events_df["NormDesc"] = events_df["Description"].apply(normalize_desc)
events_df["TextToken"] = events_df["OpName"].astype(str) + " : " + events_df["NormDesc"]
# events_df["TextToken"] = events_df["OpName"].astype(str)

# Group events by TaskID and make a sequence for each trace (TextToken)
trace_sequences = (
    events_df
    .sort_values(["TaskID", "TID"])
    .groupby("TaskID")["TextToken"]
    .apply(lambda s: " ".join(s))
    .reset_index()
    .rename(columns={"TextToken": "seq"})
)

# Merge with trace labels
traces_df["IsAbnormal"] = traces_df["IsAbnormal"].astype(int)
data = pd.merge(traces_df[["TaskID", "IsAbnormal"]], trace_sequences, on="TaskID", how="inner")

print(f"Total traces with sequences: {len(data)}")

X_text = data["seq"].tolist()
y = data["IsAbnormal"].values
task_ids = data["TaskID"].values

# =========================
# Stratified split
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_text, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =========================
# Vectorization (Bag-of-OpName + Description)
# =========================
print("\nFitting CountVectorizer on OpName + Description sequences...")
vec = CountVectorizer(token_pattern=r"\S+")
X_train_vec = vec.fit_transform(X_train)
X_val_vec   = vec.transform(X_val)
X_test_vec  = vec.transform(X_test)

print(f"Feature dimension: {X_train_vec.shape[1]}")

# =========================
# Model: Logistic Regression
# =========================
print("\nTraining logistic regression baseline...")
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # important if anomalies are rare
    n_jobs=-1,
)
clf.fit(X_train_vec, y_train)

# =========================
# Evaluation
# =========================
def eval_split(name, X, y, model):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    print(f"{name:5} | acc {acc:.3f} | pr {p:.3f} rc {r:.3f} f1 {f1:.3f}")
    return acc, p, r, f1

print("\n=== Dumb baseline results (Bag-of-OpName + LogisticRegression) ===")
eval_split("Train", X_train_vec, y_train, clf)
eval_split("Val",   X_val_vec,   y_val,   clf)
eval_split("Test",  X_test_vec,  y_test,  clf)

print("\nDetailed classification report on TEST:")
print(classification_report(y_test, clf.predict(X_test_vec), digits=3))
