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
# Dataset
# =========================
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["EventText"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(row["IsAbnormal"], dtype=torch.float)
        return item


# =========================
# Model
# =========================
class BertBinaryClassifier(nn.Module):
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
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.mlp(cls).squeeze(-1)
        return logits


# =========================
# Eval loop
# =========================
def eval_loop(model, loader, device, desc="Eval"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("========== Loading CSV sequences ==========")
    seq_df = pd.read_csv("../bert_sequences/HDFS_encoder_seq.csv")
    print(f"Loaded sequences: {len(seq_df):,}")

    df = seq_df[["EventText", "IsAbnormal"]].dropna().reset_index(drop=True)
    df["IsAbnormal"] = df["IsAbnormal"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["IsAbnormal"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["IsAbnormal"], random_state=42
    )

    print(f"Train size: {len(train_df):,}")
    print(f"Val   size: {len(val_df):,}")
    print(f"Test  size: {len(test_df):,}")

    print("\nLabel distribution:")
    print("Train:", train_df["IsAbnormal"].value_counts().to_dict())
    print("Val  :", val_df["IsAbnormal"].value_counts().to_dict())
    print("Test :", test_df["IsAbnormal"].value_counts().to_dict())

    print("========== Building datasets ==========")
    train_ds = TextDataset(train_df, tokenizer)
    val_ds   = TextDataset(val_df, tokenizer)
    test_ds  = TextDataset(test_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)
    test_loader  = DataLoader(test_ds, batch_size=32)

    print("========== Initializing model ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = BertBinaryClassifier(model_name).to(device)

    num_pos = (train_df["IsAbnormal"] == 1).sum()
    num_neg = (train_df["IsAbnormal"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    print(f"Class weights -> pos_weight={float(pos_weight):.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print("========== Training ==========")
    EPOCHS = 3

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
        val_acc, val_p, val_r, val_f1 = eval_loop(model, val_loader, device, desc=f"Epoch {epoch} [val]")

        print(
            f"\nEpoch {epoch} DONE ─ "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={val_acc:.3f} pr={val_p:.3f} rc={val_r:.3f} f1={val_f1:.3f}\n"
        )

    print("========== Final Test ==========")
    te_acc, te_p, te_r, te_f1 = eval_loop(model, test_loader, device, desc="Test")
    print(f"\nTEST → acc {te_acc:.3f}  pr {te_p:.3f}  rc {te_r:.3f}  f1 {te_f1:.3f}")
