import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
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
        item["labels"] = torch.tensor(row["label_id"], dtype=torch.long)
        return item


# =========================
# Model (multi-class)
# =========================
class BertMultiClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        print("Loading BERT backbone…")
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS]
        logits = self.mlp(cls)                 # (batch, num_classes)
        return logits


# =========================
# Eval loop
# =========================
def eval_loop(model, loader, device, desc="Eval", average="macro"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)   # (batch,)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)         # (batch,)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    return acc, p, r, f1


# =========================
# Main script
# =========================
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("========== Loading CSV ==========")
    seq_df = pd.read_csv("../bert_sequences/TB_encoder_seq.csv")
    print(f"Taille originale : {len(seq_df):,}")

    seq_df_clean = seq_df[(seq_df["FaultType"].notna())].copy()
    print(f"Taille nettoyée  : {len(seq_df_clean):,}")

    # Mapping FaultType -> label_id
    print("========== Building label mapping ==========")
    labels = sorted(seq_df_clean["FaultType"].unique())
    num_classes = len(labels)
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    print("FaultType -> id mapping:", label_to_id)

    seq_df_clean["label_id"] = seq_df_clean["FaultType"].map(label_to_id)

    # Stratified 70 / 15 / 15 split
    train_df, temp_df = train_test_split(
        seq_df_clean,
        test_size=0.30,
        stratify=seq_df_clean["FaultType"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["FaultType"],
        random_state=42,
    )

    print(f"Train size: {len(train_df):,}")
    print(f"Val   size: {len(val_df):,}")
    print(f"Test  size: {len(test_df):,}")

    print("\nLabel distribution (FaultType):")
    print("Train:\n", train_df["FaultType"].value_counts(), "\n")
    print("Val:\n",   val_df["FaultType"].value_counts(), "\n")
    print("Test:\n",  test_df["FaultType"].value_counts(), "\n")

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
    model = BertMultiClassifier(model_name, num_classes=num_classes).to(device)

    # Class weights for multi-class
    y_train = train_df["label_id"].to_numpy()
    classes = np.unique(y_train)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    print("Class weights:", class_weights_np)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print("========== Training ==========")
    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # (batch,)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_batch_loss = total_loss / (len(train_pbar))  # approx, for display
            train_pbar.set_postfix({"loss": f"{avg_batch_loss:.4f}"})

        avg_loss = total_loss / len(train_loader)
        val_acc, val_p, val_r, val_f1 = eval_loop(model, val_loader, device, desc=f"Epoch {epoch} [val]")

        print(
            f"\nEpoch {epoch} DONE ─ "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={val_acc:.3f} pr={val_p:.3f} rc={val_r:.3f} f1={val_f1:.3f}\n"
        )

    print("========== Final Test ==========")
    te_acc, te_p, te_r, te_f1 = eval_loop(model, test_loader, device, desc="Test")
    print(f"\nTEST (macro) → acc {te_acc:.3f}  pr {te_p:.3f}  rc {te_r:.3f}  f1 {te_f1:.3f}")
