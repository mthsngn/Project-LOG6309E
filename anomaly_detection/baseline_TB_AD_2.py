import os
import numpy as np
import pandas as pd
import re

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
NPZ_PATH = "../bert_cache/tb_desc_bert_bin.npz"
SEED = 42


# =========================
# Data loading
# =========================
def load_csvs(csv_path):
    traces_df = pd.read_csv(os.path.join(csv_path, "traces.csv"))
    events_df = pd.read_csv(os.path.join(csv_path, "events.csv"))
    return traces_df, events_df


print("Loading CSVs…")
traces_df, events_df = load_csvs(CSV_PATH)

events_df["TaskID"] = events_df["TaskID"].astype(str)
traces_df["TaskID"] = traces_df["TaskID"].astype(str)
traces_df["IsAbnormal"] = traces_df["IsAbnormal"].astype(int)


# =========================
# Load BERT .npz embeddings
# =========================
print(f"Loading BERT embeddings from {NPZ_PATH}")
npz = np.load(NPZ_PATH)
bert_embs = npz["embs"]          # shape (N_events, H)
hidden_size = bert_embs.shape[1]

events_df = events_df.reset_index(drop=True)
assert len(events_df) == bert_embs.shape[0], "Mismatch events_df vs npz embeddings"

events_df["desc_emb"] = list(bert_embs)


# =========================
# Compute per-trace embeddings (mean)
# =========================
print("Computing per-trace BERT embeddings…")

trace_vecs = []
trace_labels = []
trace_ids = []

for tid, group in events_df.groupby("TaskID"):
    mat = np.stack(group["desc_emb"].to_list())   # (num_events, H)
    emb = mat.mean(axis=0)                        # (H,)
    
    trace_vecs.append(emb)
    trace_labels.append(
        int(traces_df.loc[traces_df["TaskID"] == tid, "IsAbnormal"].iloc[0])
    )
    trace_ids.append(tid)

X = np.vstack(trace_vecs)     # (num_traces, H)
y = np.array(trace_labels)

print("Trace-level embeddings shape:", X.shape)


# =========================
# Stratified split
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# =========================
# MLPClassifier
# =========================
print("\nTraining MLPClassifier baseline...")

clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,            # L2
    batch_size=64,
    learning_rate="adaptive",
    max_iter=200,
    random_state=SEED,
    verbose=False,
)

clf.fit(X_train, y_train)


# =========================
# Evaluation
# =========================
def eval_split(name, X, y, model):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    print(f"{name:5} | acc {acc:.3f} | pr {p:.3f} rc {r:.3f} f1 {f1:.3f}")
    return acc, p, r, f1


print("\n=== Baseline: BERT mean-pooled per trace + MLPClassifier ===")
eval_split("Train", X_train, y_train, clf)
eval_split("Val",   X_val,   y_val,   clf)
eval_split("Test",  X_test,  y_test,  clf)

print("\nClassification report (TEST):")
print(classification_report(y_test, clf.predict(X_test), digits=3))
