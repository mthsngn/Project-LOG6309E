import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)
from sklearn.neural_network import MLPClassifier

# =========================
# Config
# =========================
CSV_PATH = "../master_tables/TB/test"
NPZ_PATH = "../bert_cache/tb_desc_bert_multi.npz"
SEED = 42

# =========================
# Data loading
# =========================
def load_csvs(csv_path):
    traces_df = pd.read_csv(os.path.join(csv_path, "traces.csv"))
    events_df = pd.read_csv(os.path.join(csv_path, "events.csv"))
    return traces_df, events_df

traces_df, events_df = load_csvs(CSV_PATH)

traces_df["TaskID"] = traces_df["TaskID"].astype(str)
events_df["TaskID"] = events_df["TaskID"].astype(str)

# =========================
# Keep only labeled traces
# =========================
valid_traces = traces_df[traces_df["FaultType"].notna()].copy()
valid_ids = valid_traces["TaskID"].astype(str).tolist()

traces_df = traces_df[traces_df["TaskID"].isin(valid_ids)].reset_index(drop=True)
events_df = events_df[events_df["TaskID"].isin(valid_ids)].reset_index(drop=True)

# =========================
# Load BERT npz embeddings (must match this filtered events_df)
# =========================
print(f"Loading BERT embeddings from {NPZ_PATH}")
npz = np.load(NPZ_PATH)
embs = npz["embs"]                  # (N_events_filtered, H)
hidden_dim = embs.shape[1]

assert len(events_df) == embs.shape[0], (
    f"Mismatch events vs embeddings: events_df={len(events_df)}, embs={embs.shape[0]}"
)

events_df["desc_emb"] = list(embs)

# =========================
# Label dataframe & encoding (FaultType -> id)
# =========================
label_df = traces_df[["TaskID", "FaultType"]].copy()
labels = sorted(label_df["FaultType"].unique())
label_to_id = {lab: i for i, lab in enumerate(labels)}
print("FaultType -> id mapping:", label_to_id)

label_df["label_id"] = label_df["FaultType"].map(label_to_id)

# =========================
# Build per-trace BERT vectors (mean pooling)
# =========================
print("Building per-trace BERT vectors…")

trace_vecs = []
trace_labels = []

for tid, group in events_df.groupby("TaskID"):
    mat = np.stack(group["desc_emb"].tolist())   # (num_events, H)
    trace_emb = mat.mean(axis=0)                 # (H,)
    trace_vecs.append(trace_emb)

    lbl = int(label_df.loc[label_df["TaskID"] == tid, "label_id"].iloc[0])
    trace_labels.append(lbl)

X = np.vstack(trace_vecs)
y = np.array(trace_labels)

print("Final trace embedding shape:", X.shape)
print("Num classes:", len(labels))

# =========================
# Stratified split (multi-class)
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED,
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =========================
# Model: Multiclass MLPClassifier
# =========================
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,            # L2 regularization
    batch_size=64,
    learning_rate="adaptive",
    max_iter=200,
    random_state=SEED,
    verbose=False,
)

print("Training multiclass MLP on BERT embeddings…")
clf.fit(X_train, y_train)

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

print("\n=== Multiclass baseline (BERT Description mean-pooled + MLPClassifier) ===")
eval_split("Train", X_train, y_train, clf)
eval_split("Val",   X_val,   y_val,   clf)
print("\n--- TEST ---")
eval_split("Test",  X_test,  y_test,  clf, target_names=labels)
