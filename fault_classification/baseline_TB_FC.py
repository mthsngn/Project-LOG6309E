import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)

# =========================
# Config
# =========================
CSV_PATH = "../master_tables/HDFS/test"
SEED = 42

# =========================
# Data loading
# =========================
def load_csvs(csv_path):
    traces_df = pd.read_csv(os.path.join(csv_path, "traces.csv"))
    events_df = pd.read_csv(os.path.join(csv_path, "events.csv"))
    return traces_df, events_df

traces_df, events_df = load_csvs(CSV_PATH)

print("Building per-trace OpName + Description sequences for multi-class FaultType...")

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
label_df = traces_df[["TaskID", "FaultType"]].copy()
label_df = label_df[label_df["FaultType"].notna()]
data = pd.merge(label_df, trace_sequences, on="TaskID", how="inner")

print(f"Total traces with sequences & FaultType: {len(data)}")

# =========================
# Encode labels (FaultType -> id)
# =========================
labels = sorted(data["FaultType"].unique())
num_classes = len(labels)
label_to_id = {lab: i for i, lab in enumerate(labels)}
print("FaultType -> id mapping:", label_to_id)

data["label_id"] = data["FaultType"].map(label_to_id)

# =========================
# Stratified split (multi-class)
# =========================
X_text = data["seq"].tolist()
y = data["label_id"].values
X_train, X_temp, y_train, y_temp = train_test_split(
    X_text, y,
    test_size=0.30,
    stratify=y,
    random_state=SEED,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED,
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =========================
# Vectorization (Bag-of-OpName + Description)
# =========================
print("\nFitting CountVectorizer on OpName + Description sequences...")
vec = CountVectorizer(token_pattern=r"\S+", min_df=5)
X_train_vec = vec.fit_transform(X_train)
X_val_vec   = vec.transform(X_val)
X_test_vec  = vec.transform(X_test)

print(f"Feature dimension: {X_train_vec.shape[1]}")

# =========================
# Model: Multiclass Logistic Regression
# =========================
print("\nTraining multiclass logistic regression baseline...")

clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    n_jobs=-1,
    multi_class="auto",   # for multiclass
    solver="lbfgs",
)
clf.fit(X_train_vec, y_train)

# =========================
# Evaluation helpers
# =========================
def eval_split(name, X, y, model, target_names=None):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    print(f"{name:5} | acc {acc:.3f} | macro_pr {p_macro:.3f} rc {r_macro:.3f} f1 {f1_macro:.3f}")
    if target_names is not None:
        print("\nDetailed classification report:")
        print(classification_report(y, y_pred, target_names=target_names, digits=3))
    return acc, p_macro, r_macro, f1_macro

print("\n=== Multiclass baseline results (Bag-of-OpName+Desc + LogisticRegression) ===")
eval_split("Train", X_train_vec, y_train, clf)
eval_split("Val",   X_val_vec,   y_val,   clf)
print("\n--- TEST ---")
eval_split("Test",  X_test_vec,  y_test,  clf, target_names=labels)
