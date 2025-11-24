import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import hashlib
import json


# =========================
# Utilities
# =========================
def _graphs_cache_path(cache_dir, split_name, ids):
    os.makedirs(cache_dir, exist_ok=True)
    s = json.dumps(list(map(str, ids)), separators=(",", ":"), ensure_ascii=False)
    fid = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{split_name}_graphs_{fid}.pt")

def stats(graphs):
    ns = [g.x.size(0) for g in graphs]
    es = [g.edge_index.size(1) for g in graphs]
    return np.mean(ns), np.mean(es)

def show_split_distribution_multiclass(traces_df, name, ids, label_col="FaultType", top_k=5):
    subset = traces_df[traces_df["TaskID"].astype(str).isin(ids)]
    vc = subset[label_col].astype(str).value_counts()
    total = int(vc.sum())
    head = ", ".join([f"{k}:{int(v)}({v/total:.1%})" for k, v in vc.head(top_k).items()])
    print(f"{name:<5} : total={total:5d} | classes={len(vc)} | top{top_k}: {head}")

# =========================
# Data loading & splitting
# =========================
def load_csvs(csv_path):
    traces_df = pd.read_csv(os.path.join(csv_path, "traces.csv"))
    events_df = pd.read_csv(os.path.join(csv_path, "events.csv"))
    edges_df  = pd.read_csv(os.path.join(csv_path, "edges.csv"))
    ops_df    = pd.read_csv(os.path.join(csv_path, "operations.csv"))
    return traces_df, events_df, edges_df, ops_df

def stratified_ids(traces_df, seed=42, label_col="FaultType"):
    trace_ids = traces_df["TaskID"].astype(str)
    labels    = traces_df[label_col].astype(str)
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        trace_ids, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

# =========================
# Graph building
# =========================
def build_op_vocab(events_df):
    unique_ops = sorted(events_df["OpName"].dropna().astype(str).unique())
    opname_to_ix = {op: i + 1 for i, op in enumerate(unique_ops)}
    opname_to_ix["<UNK>"] = 0
    num_ops = len(opname_to_ix)
    return opname_to_ix, num_ops

def build_fault_vocab(traces_df, label_col="FaultType"):
    labels = traces_df[label_col].astype(str).fillna("<UNK>").unique()
    labels = sorted(labels)
    fault_to_ix = {lab: i for i, lab in enumerate(labels)}
    ix_to_fault = {i: lab for lab, i in fault_to_ix.items()}
    num_classes = len(fault_to_ix)
    return fault_to_ix, ix_to_fault, num_classes

def build_graph_data_for_trace(events_df, edges_df, traces_df, task_id, opname_to_ix=None,
                               label_col="FaultType", fault_to_ix=None):
    ev = events_df[events_df["TaskID"] == task_id].copy()
    ed = edges_df[edges_df["TaskID"] == task_id].copy()
    tr = traces_df[traces_df["TaskID"] == task_id].iloc[0]

    tids = ev["TID"].astype(str).tolist()
    idx = {t: i for i, t in enumerate(tids)}

    indeg = pd.Series(0, index=tids); outdeg = pd.Series(0, index=tids)
    for _, r in ed.iterrows():
        if r["FatherTID"] in idx: outdeg[r["FatherTID"]] += 1
        if r["ChildTID"]  in idx: indeg[r["ChildTID"]]  += 1

    indeg_d, outdeg_d = indeg.to_dict(), outdeg.to_dict()
    x_num = np.c_[ev["TID"].map(indeg_d).fillna(0).to_numpy(), ev["TID"].map(outdeg_d).fillna(0).to_numpy()].astype("float32")

    op_ix = None
    if opname_to_ix is not None and "OpName" in ev.columns:
        op_ix = ev["OpName"].map(lambda s: opname_to_ix.get(str(s), 0)).to_numpy().astype("int64")

    if fault_to_ix is None:
        raise ValueError("fault_to_ix mapping required for FaultType classification.")
    y_id = fault_to_ix.get(str(tr[label_col]), None)
    if y_id is None:
        y_id = fault_to_ix.get("<UNK>", 0)

    src, dst = [], []
    for _, r in ed.iterrows():
        u, v = r["FatherTID"], r["ChildTID"]
        if u in idx and v in idx:
            src.append(idx[u]); dst.append(idx[v])

    data = {
        "x_num": x_num,
        "edge_index": np.array([src, dst], dtype="int64"),
        "y": np.array([int(y_id)], dtype="int64"),
        "TraceId": task_id,
    }
    if op_ix is not None:
        data["op_idx"] = op_ix
    return data


def to_pyg_data(g):
    x_num = torch.from_numpy(g["x_num"])
    op_idx = torch.from_numpy(g["op_idx"]).long() if "op_idx" in g else None
    edge_index = torch.from_numpy(g["edge_index"]).long()
    y = torch.from_numpy(g["y"]).long()
    data = Data(x=x_num, edge_index=edge_index, y=y)
    if op_idx is not None:
        data.op_idx = op_idx
    data.trace_id = g["TraceId"]
    return data

def build_graphs(ids, split_name, events_df, edges_df, traces_df, opname_to_ix,
                 fault_to_ix, label_col="FaultType", progress_every=50,
                 cache_dir="graph_cache"):
    # ----- try cache -----
    cache_path = _graphs_cache_path(cache_dir, split_name, ids)
    if os.path.exists(cache_path):
        print(f"\nLoading cached {split_name} graphs from {cache_path} ...")
        return torch.load(cache_path)

    # ----- build from scratch -----
    print(f"\nBuilding {split_name} graphs ({len(ids)} traces)...")
    graphs, skipped = [], 0
    for i, tid in enumerate(ids, 1):
        try:
            g = build_graph_data_for_trace(
                events_df, edges_df, traces_df, tid,
                opname_to_ix=opname_to_ix,
                label_col=label_col,
                fault_to_ix=fault_to_ix
            )
            graphs.append(to_pyg_data(g))
        except Exception as e:
            skipped += 1
            print(f"Skipping {tid}: {e}")
        if i % progress_every == 0 or i == len(ids):
            print(f"Processed {i}/{len(ids)} | valid: {len(graphs)} | skipped: {skipped}")

    print(f"Done {split_name}: {len(graphs)} valid, {skipped} skipped")

    # ----- save cache -----
    torch.save(graphs, cache_path)
    print(f"Saved {split_name} graphs to {cache_path}\n")
    return graphs


# =========================
# Model & training
# =========================
class GraphClassifier(nn.Module):
    def __init__(self, num_ops, x_num_dim=2, emb_dim=16, hidden=64, num_classes=2):
        super().__init__()
        self.op_emb = nn.Embedding(num_ops, emb_dim)
        in_dim = x_num_dim + emb_dim
        self.gcn1 = GCNConv(in_dim, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.lin  = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x_num = data.x
        op_idx = getattr(data, "op_idx", None)
        if op_idx is not None:
            x_cat = self.op_emb(op_idx)
        else:
            x_cat = torch.zeros(x_num.size(0), 16, device=x_num.device)
        x = torch.cat([x_num, x_cat], dim=1)
        x = self.gcn1(x, data.edge_index).relu()
        x = self.gcn2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        logits = self.lin(x)
        return logits

def run_epoch(model, loader, device, opt, criterion, training=True):
    if training: model.train()
    else:        model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
    return loss_sum / total, correct / total

def prf_metrics(model, loader, device, average="macro"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return acc, p, r, f1

def make_loaders(train_graphs, val_graphs, test_graphs, batch_size=16):
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size)
    return train_loader, val_loader, test_loader

def init_model(train_graphs, num_ops, device, num_classes, lr=1e-3, wd=1e-4, class_weights=None):
    x_num_dim = train_graphs[0].x.size(1)
    model = GraphClassifier(num_ops=num_ops, x_num_dim=x_num_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return model, opt, criterion

def train_and_eval(model, opt, criterion, train_loader, val_loader, device, epochs=20):
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt, criterion, training=True)
        va_loss, va_acc = run_epoch(model, val_loader,   device, opt, criterion, training=False)
        va_acc_m, va_p, va_r, va_f1 = prf_metrics(model, val_loader, device, average="macro")
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
            f"VAL (macro) pr {va_p:.3f} rc {va_r:.3f} f1 {va_f1:.3f}"
        )

def test_model(model, test_loader, device):
    te_acc, te_p, te_r, te_f1 = prf_metrics(model, test_loader, device)
    print(f"\nTEST  acc {te_acc:.3f} | pr {te_p:.3f} rc {te_r:.3f} f1 {te_f1:.3f}")

# =========================
# Main
# =========================

if __name__ == "__main__":
    CSV_PATH = "../master_tables/HDFS/test"
    SEED = 42; 
    BATCH_SIZE = 16; 
    EPOCHS = 20
    LABEL_COL = "FaultType"

    traces_df, events_df, edges_df, ops_df = load_csvs(CSV_PATH)
    traces_df = traces_df[traces_df["IsAbnormal"] == 1].reset_index(drop=True)

    # Stratify by FaultType
    train_ids, val_ids, test_ids = stratified_ids(traces_df, seed=SEED, label_col=LABEL_COL)

    print(f"Total traces: {len(traces_df['TaskID'].unique())}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print("\nClass distribution (top classes):")
    show_split_distribution_multiclass(traces_df, "Train", train_ids, label_col=LABEL_COL)
    show_split_distribution_multiclass(traces_df, "Val",   val_ids,   label_col=LABEL_COL)
    show_split_distribution_multiclass(traces_df, "Test",  test_ids,  label_col=LABEL_COL)

    # Vocabularies
    opname_to_ix, num_ops = build_op_vocab(events_df)
    fault_to_ix, ix_to_fault, num_classes = build_fault_vocab(traces_df, label_col=LABEL_COL)

    # Graphs
    train_graphs = build_graphs(train_ids, "train", events_df, edges_df, traces_df,
                                opname_to_ix, fault_to_ix, label_col=LABEL_COL)
    val_graphs   = build_graphs(val_ids, "val", events_df, edges_df, traces_df,
                                opname_to_ix, fault_to_ix, label_col=LABEL_COL)
    test_graphs  = build_graphs(test_ids, "test", events_df, edges_df, traces_df,
                                opname_to_ix, fault_to_ix, label_col=LABEL_COL)

    train_loader, val_loader, test_loader = make_loaders(train_graphs, val_graphs, test_graphs, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = np.array([g.y.item() for g in train_graphs])
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    model, opt, criterion = init_model(
    train_graphs, num_ops, device, num_classes, class_weights=class_weights
    )

    # Train / Eval
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt, criterion, training=True)
        va_loss, va_acc = run_epoch(model, val_loader,   device, opt, criterion, training=False)
        va_acc_m, va_p, va_r, va_f1 = prf_metrics(model, val_loader, device, average="macro")
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
            f"VAL (macro) pr {va_p:.3f} rc {va_r:.3f} f1 {va_f1:.3f}"
        )

    te_acc, te_p, te_r, te_f1 = prf_metrics(model, test_loader, device, average="macro")
    print(f"\nTEST (macro) acc {te_acc:.3f} | pr {te_p:.3f} rc {te_r:.3f} f1 {te_f1:.3f}")

