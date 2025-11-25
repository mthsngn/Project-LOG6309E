import os
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm


# =========================
# Utils
# =========================

def split_events(text: str):
    """
    Split the EventText of a session into individual event strings.
    """
    events = [e.strip() for e in text.split("[SEP]") if e.strip()]
    if not events:
        # fallback: if for some reason there is no separator
        events = [text.strip()] if text.strip() else [""]
    return events


# =========================
# Dataset (session-level, list of events)
# =========================
class SessionDataset(Dataset):
    """
    Each item is ONE session / TaskID.

    Columns expected in df:
        - TaskID
        - EventText: concatenated events for that task
        - IsAbnormal: 0/1 label

    We:
        - split EventText into events,
        - optionally cap to MAX_EVENTS (keep last ones),
        - tokenize all events (list of strings),
        - return input_ids, attention_mask with shape (max_events, seq_len).
    """

    def __init__(self, df, tokenizer, max_len=128, max_events=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_events = max_events

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        event_text = str(row["EventText"])
        label = int(row["IsAbnormal"])
        task_id = int(row["TaskID"])

        # 1) Split session into individual events
        events = split_events(event_text)

        # 2) Truncate / keep last max_events
        if len(events) > self.max_events:
            events = events[-self.max_events:]

        # 3) Pad with empty strings if too short
        if len(events) < self.max_events:
            events = events + [""] * (self.max_events - len(events))

        # 4) Tokenize list of events
        enc = self.tokenizer(
            events,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",   # shapes: (max_events, seq_len)
        )

        item = {k: v for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        item["task_id"] = torch.tensor(task_id, dtype=torch.long)
        return item


# =========================
# Model: BERT → event CLS → mean over events → MLP
# =========================
class BertSessionClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        print("Loading BERT backbone…")
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        """
        input_ids:      (batch, max_events, seq_len)
        attention_mask: (batch, max_events, seq_len)
        """
        b, n, L = input_ids.shape

        # Flatten events across batch: (b*n, L)
        input_ids = input_ids.view(b * n, L)
        attention_mask = attention_mask.view(b * n, L)

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (b*n, hidden)

        # Reshape back to (batch, max_events, hidden)
        hidden = cls.view(b, n, -1)

        # Mean pooling over events -> session embedding
        session_emb = hidden.mean(dim=1)       # (batch, hidden)

        logits = self.mlp(session_emb).squeeze(-1)  # (batch,)
        return logits


# =========================
# Eval loop (session-level directly)
# =========================
def eval_loop(model, loader, device, desc="Eval"):
    """
    Evaluate at TASK (session) level.

    Each batch item is already a full session, so we
    just compute metrics over the session predictions.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)          # (B, max_events, L)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().long())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return acc, p, r, f1


# =========================
# Main script
# =========================
if __name__ == "__main__":

    model_name = "bert-base-uncased"
    MAX_LEN = 128      # tokens per event
    MAX_EVENTS = 64    # events per session (cap)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("========== Loading CSV sequences ==========")
    seq_df = pd.read_csv("../bert_sequences/HDFS_encoder_seq.csv")
    print(f"Loaded sequences (tasks/sessions): {len(seq_df):,}")

    # Base sessions dataframe
    base_df = seq_df[["TaskID", "EventText", "IsAbnormal"]].dropna().reset_index(drop=True)
    base_df["IsAbnormal"] = base_df["IsAbnormal"].astype(int)
    base_df["TaskID"] = base_df["TaskID"].astype(int)

    print("========== Splitting tasks (train/val/test) ==========")
    train_sess_df, temp_sess_df = train_test_split(
        base_df,
        test_size=0.30,
        stratify=base_df["IsAbnormal"],
        random_state=42,
    )
    val_sess_df, test_sess_df = train_test_split(
        temp_sess_df,
        test_size=0.50,
        stratify=temp_sess_df["IsAbnormal"],
        random_state=42,
    )

    print(f"Train tasks: {len(train_sess_df):,}")
    print(f"Val   tasks: {len(val_sess_df):,}")
    print(f"Test  tasks: {len(test_sess_df):,}")

    print("\nTask label distribution:")
    print("Train:", train_sess_df["IsAbnormal"].value_counts().to_dict())
    print("Val  :", val_sess_df["IsAbnormal"].value_counts().to_dict())
    print("Test :", test_sess_df["IsAbnormal"].value_counts().to_dict())

    print("\n========== Building datasets (session-level) ==========")
    train_ds = SessionDataset(train_sess_df, tokenizer, max_len=MAX_LEN, max_events=MAX_EVENTS)
    val_ds   = SessionDataset(val_sess_df,   tokenizer, max_len=MAX_LEN, max_events=MAX_EVENTS)
    test_ds  = SessionDataset(test_sess_df,  tokenizer, max_len=MAX_LEN, max_events=MAX_EVENTS)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)   # smaller B because heavier
    val_loader   = DataLoader(val_ds,   batch_size=8)
    test_loader  = DataLoader(test_ds,  batch_size=8)

    print("========== Initializing model ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = BertSessionClassifier(model_name).to(device)

    # Class weights computed on sessions
    num_pos = (train_sess_df["IsAbnormal"] == 1).sum()
    num_neg = (train_sess_df["IsAbnormal"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    print(f"Class weights -> pos_weight={float(pos_weight):.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print("========== Training ==========")
    EPOCHS = 15

    best_f1 = -1.0
    best_epoch = -1
    best_model_path = "best_HDFS_bert.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_acc, val_p, val_r, val_f1 = eval_loop(
            model, val_loader, device, desc=f"Epoch {epoch} [val]"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"--> New best model saved (epoch {epoch}, val_f1={val_f1:.3f})")

        print(
            f"\nEpoch {epoch} DONE ─ "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={val_acc:.3f} pr={val_p:.3f} rc={val_r:.3f} f1={val_f1:.3f}\n"
        )

    print(f"\nLoading best model from epoch {best_epoch} (val_f1={best_f1:.3f})...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("========== Final Test ==========")
    te_acc, te_p, te_r, te_f1 = eval_loop(model, test_loader, device, desc="Test")
    print(f"\nTEST (Task-level) → acc {te_acc:.3f}  pr {te_p:.3f}  rc {te_r:.3f}  f1 {te_f1:.3f}")
