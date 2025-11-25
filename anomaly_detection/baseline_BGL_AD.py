import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# =========================
# Config
# =========================
CSV_PATH = "../master_tables/BGL/test"
SEED = 42

# =========================
# Data loading
# =========================
def load_csvs(csv_path):
    traces_df = pd.read_csv(os.path.join(csv_path, "traces.csv"))
    events_df = pd.read_csv(os.path.join(csv_path, "events.csv"))
    edges_df  = pd.read_csv(os.path.join(csv_path, "edges.csv"))
    return traces_df, events_df, edges_df

traces_df, events_df, _ = load_csvs(CSV_PATH)

print("Building EventId sequences...")

events_df["TaskID"] = events_df["TaskID"].astype(int)
traces_df["TaskID"] = traces_df["TaskID"].astype(int)

# Group events by TaskID and make a sequence for each trace (list of EventIds)
trace_sequences = (
    events_df
    .sort_values(["TaskID", "TID"])
    .groupby("TaskID")["EventId"]
    .apply(lambda s: " ".join(map(str, s)))
    .reset_index()
    .rename(columns={"EventId": "seq"})
)
# Merge with trace labels
traces_df["IsAbnormal"] = traces_df["IsAbnormal"].astype(int)
data = pd.merge(traces_df[["TaskID", "IsAbnormal"]], trace_sequences, on="TaskID", how="inner")

print(f"Total traces with sequences: {len(data)}")

# =========================
# Stratified split
# =========================
X_text = data["seq"].tolist()
y = data["IsAbnormal"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X_text, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =========================
# Vectorization (Bag-of-EventId = MCV)
# =========================
print("\nFitting CountVectorizer on EventId sequences...")
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
    max_iter=3000,
    class_weight="balanced",
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

print("\n=== Dumb baseline results (Bag-of-EventId + LogisticRegression) ===")
eval_split("Train", X_train_vec, y_train, clf)
eval_split("Val",   X_val_vec,   y_val,   clf)
eval_split("Test",  X_test_vec,  y_test,  clf)

print("\nDetailed classification report on TEST:")
print(classification_report(y_test, clf.predict(X_test_vec), digits=3))
