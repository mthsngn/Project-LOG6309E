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
    # fingerprint depends on the exact ordered TaskID list
    s = json.dumps(list(map(str, ids)), separators=(",", ":"), ensure_ascii=False)
    fid = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{split_name}_graphs_{fid}.pt")

def stats(graphs):
    ns = [g.x.size(0) for g in graphs]
    es = [g.edge_index.size(1) for g in graphs]
    return np.mean(ns), np.mean(es)

def show_split_distribution(traces_df, name, ids):
    subset = traces_df[traces_df["TaskID"].isin(ids)]
    counts = subset["IsAbnormal"].value_counts().to_dict()
    total = len(subset)
    normal = counts.get(0, 0); abnormal = counts.get(1, 0)
    print(f"{name:<5} : total={total:5d} | normal={normal:5d} ({normal/total:.2%}) | abnormal={abnormal:5d} ({abnormal/total:.2%})")

# =========================
# Data loading & splitting
# =========================
def load_csvs(csv_path):
    traces_df = pd.read_csv(os.path.join(csv_path, "traces.csv"))
    events_df = pd.read_csv(os.path.join(csv_path, "events.csv"))
    edges_df  = pd.read_csv(os.path.join(csv_path, "edges.csv"))
    return traces_df, events_df, edges_df

def stratified_ids(traces_df, seed=42):
    trace_ids = traces_df["TaskID"]
    labels    = traces_df["IsAbnormal"].astype(int)
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
    unique_eids = sorted(events_df["EventId"].dropna().astype(str).unique())
    eid_to_ix = {eid: i + 1 for i, eid in enumerate(unique_eids)}
    eid_to_ix["<UNK>"] = 0
    num_tokens = len(eid_to_ix)
    return eid_to_ix, num_tokens

def build_graph_data_for_trace(events_df, edges_df, traces_df, task_id, eid_to_ix=None):
    # Filter rows for this trace
    ev = events_df[events_df["TaskID"] == task_id].copy()
    ed = edges_df[edges_df["TaskID"] == task_id].copy()
    tr = traces_df[traces_df["TaskID"] == task_id].iloc[0]

    # Build Timestamp from Date + Time
    try:
        date_str = ev["Date"].astype(str).str.zfill(6)   # YYMMDD
        time_str = ev["Time"].astype(str).str.zfill(6)   # HHMMSS

        dt = pd.to_datetime(
            date_str + time_str,
            format="%y%m%d%H%M%S",
            errors="coerce"
        )
        ev["Timestamp"] = (dt.astype("int64") // 10**9).astype("float64")
    except Exception:
        # fallback if parsing fails : monotonic pseudo-time
        ev["Timestamp"] = ev["TID"].astype(float)

    # Map TID -> local index in [0..n-1]
    tids = ev["TID"].astype(str).tolist()
    idx = {t: i for i, t in enumerate(tids)}

    # --------------------------------------
    # 1) DEGREE-BASED FEATURES
    # --------------------------------------
    indeg = pd.Series(0, index=tids)
    outdeg = pd.Series(0, index=tids)

    for _, r in ed.iterrows():
        father = str(r["FatherTID"])
        child  = str(r["ChildTID"])
        if father in idx:
            outdeg[father] += 1
        if child in idx:
            indeg[child] += 1

    indeg_d, outdeg_d = indeg.to_dict(), outdeg.to_dict()

    # --------------------------------------
    # 2) POSITION FEATURES (pos, dist_end)
    # --------------------------------------
    num_nodes = len(ev)
    denom = max(1, num_nodes - 1)

    tid_nums = ev["TID"].astype(int).to_numpy()
    pos = tid_nums / denom
    dist_end = (denom - tid_nums) / denom

    # --------------------------------------
    # 3) TIMESTAMP FEATURES (normalized)
    # --------------------------------------
    ts = ev["Timestamp"].astype("float64").to_numpy()
    tsn = (ts - ts.min()) / max(1e-9, (ts.max() - ts.min()))

    # --------------------------------------
    # CONCAT ALL NUMERIC FEATURES
    # --------------------------------------
    x_num = np.c_[
        ev["TID"].astype(str).map(indeg_d).fillna(0).to_numpy(),
        ev["TID"].astype(str).map(outdeg_d).fillna(0).to_numpy(),
        pos.astype("float32"),
        dist_end.astype("float32"),
        tsn.astype("float32"),
    ].astype("float32")

    # --------------------------------------
    # CATEGORICAL TOKEN: EventId
    # --------------------------------------
    token_idx = None
    if eid_to_ix is not None:
        token_idx = ev["EventId"].astype(str).map(
            lambda s: eid_to_ix.get(s, 0)
        ).to_numpy().astype("int64")

    # --------------------------------------
    # EDGES
    # --------------------------------------
    src, dst = [], []
    for _, r in ed.iterrows():
        u, v = str(r["FatherTID"]), str(r["ChildTID"])
        if u in idx and v in idx:
            src.append(idx[u])
            dst.append(idx[v])

    # --------------------------------------
    # FINAL GRAPH DICT
    # --------------------------------------
    data = {
        "x_num": x_num,
        "edge_index": np.array([src, dst], dtype="int64"),
        "y": np.array([int(tr["IsAbnormal"])], dtype="int64"),
        "TraceId": task_id,
    }

    if token_idx is not None:
        data["op_idx"] = token_idx

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

def build_graphs(ids, split_name, events_df, edges_df, traces_df, eid_to_ix,
                 progress_every=50, cache_dir="../graph_cache"):
    # ---- try cache first ----
    cache_path = _graphs_cache_path(cache_dir, split_name, ids)
    if os.path.exists(cache_path):
        print(f"\nLoading cached {split_name} graphs from {cache_path} ...")
        return torch.load(cache_path)

    # ---- build from scratch ----
    print(f"\nBuilding {split_name} graphs ({len(ids)} traces)...")
    graphs, skipped = [], 0
    for i, tid in enumerate(ids, 1):
        try:
            g = build_graph_data_for_trace(events_df, edges_df, traces_df, tid, eid_to_ix)
            g = to_pyg_data(g)
            if g is not None:
                graphs.append(g)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"Skipping {tid}: {e}")

        if i % progress_every == 0 or i == len(ids):
            print(f"Processed {i}/{len(ids)} | valid: {len(graphs)} | skipped: {skipped}")

    print(f"Done {split_name}: {len(graphs)} valid, {skipped} skipped")

    # ---- save cache ----
    torch.save(graphs, cache_path)
    print(f"Saved {split_name} graphs to {cache_path}\n")
    return graphs

# =========================
# Model & training
# =========================
class GraphClassifier(nn.Module):
    def __init__(self, num_ops, x_num_dim=5, emb_dim=16, hidden=64, num_classes=2):
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

def prf_metrics(model, loader, device):
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
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return acc, p, r, f1

def make_loaders(train_graphs, val_graphs, test_graphs, batch_size=16):
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size)
    return train_loader, val_loader, test_loader

def init_model(train_graphs, num_ops, device, lr=1e-3, wd=1e-4, class_weights=None):
    x_num_dim = train_graphs[0].x.size(1)
    model = GraphClassifier(num_ops=num_ops, x_num_dim=x_num_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return model, opt, criterion

def train_and_eval(model, opt, criterion, train_loader, val_loader, device, epochs=20):
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt, criterion, training=True)
        va_loss, va_acc = run_epoch(model, val_loader,   device, opt, criterion, training=False)
        va_acc_m, va_p, va_r, va_f1 = prf_metrics(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
            f"VAL pr {va_p:.3f} rc {va_r:.3f} f1 {va_f1:.3f}"
        )

def test_model(model, test_loader, device):
    te_acc, te_p, te_r, te_f1 = prf_metrics(model, test_loader, device)
    print(f"\nTEST  acc {te_acc:.3f} | pr {te_p:.3f} rc {te_r:.3f} f1 {te_f1:.3f}")

# =========================
# Main
# =========================

if __name__ == "__main__":
    CSV_PATH = "../master_tables/HDFS/test"
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 5
    
    traces_df, events_df, edges_df = load_csvs(CSV_PATH)

    train_ids, val_ids, test_ids = stratified_ids(traces_df, seed=SEED)

    print(f"Total traces: {len(traces_df['TaskID'].unique())}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print("\nClass distribution:")
    show_split_distribution(traces_df, "Train", train_ids)
    show_split_distribution(traces_df, "Val",   val_ids)
    show_split_distribution(traces_df, "Test",  test_ids)

    eid_to_ix, num_ops = build_op_vocab(events_df)

    train_graphs = build_graphs(train_ids, "train", events_df, edges_df, traces_df, eid_to_ix)
    val_graphs   = build_graphs(val_ids,   "val",   events_df, edges_df, traces_df, eid_to_ix)
    test_graphs  = build_graphs(test_ids,  "test",  events_df, edges_df, traces_df, eid_to_ix)

    train_loader, val_loader, test_loader = make_loaders(train_graphs, val_graphs, test_graphs, BATCH_SIZE)
    print("Graphs  | train:", len(train_graphs), "val:", len(val_graphs), "test:", len(test_graphs))
    print("Batches | train:", len(train_loader), "val:", len(val_loader), "test:", len(test_loader))
    mN, mE = stats(train_graphs)
    print(f"Avg nodes/graph: {mN:.1f} | Avg edges/graph: {mE:.1f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = np.array([g.y.item() for g in train_graphs])
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    model, opt, criterion = init_model(train_graphs, num_ops, device, class_weights=class_weights)

    train_and_eval(model, opt, criterion, train_loader, val_loader, device, EPOCHS)
    test_model(model, test_loader, device)
