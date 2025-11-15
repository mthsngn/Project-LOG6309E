import os
import re
from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import  torch
from sklearn.model_selection import (train_test_split)
import torch
import torch.nn as nn

from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
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
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(row["IsAbnormal"], dtype=torch.float)
        return item
#  modèle BERT + MLP
class BertBinaryClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # token [CLS]
        logits = self.mlp(cls).squeeze(-1)     # (batch,)
        return logits
def eval_loop(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
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




model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess(batch):
    return tokenizer(
        batch["EventText"],
        truncation=True,
        padding=False,  # on laissera le DataCollator gérer
        max_length=256,
    )



seq_df  = pd.read_csv('master_tables/encoder_seq.csv')
df = seq_df[["EventText", "IsAbnormal"]].dropna().reset_index(drop=True)
df["IsAbnormal"] = df["IsAbnormal"].astype(int)

# 2) split stratifié 70 / 15 / 15
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["IsAbnormal"],
    random_state=42,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["IsAbnormal"],
    random_state=42,
)

print("Train:", train_df["IsAbnormal"].value_counts())
print("Val:  ", val_df["IsAbnormal"].value_counts())
print("Test: ", test_df["IsAbnormal"].value_counts())

train_ds = TextDataset(train_df, tokenizer)
val_ds   = TextDataset(val_df, tokenizer)
test_ds  = TextDataset(test_df, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertBinaryClassifier(model_name).to(device)

# 5) loss pondérée (dataset unbalanced)
num_pos = (train_df["IsAbnormal"] == 1).sum()
num_neg = (train_df["IsAbnormal"] == 0).sum()
pos_weight = torch.tensor([num_neg / num_pos], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_acc, val_p, val_r, val_f1 = eval_loop(val_loader)
    print(f"Epoch {epoch} | train loss {total_loss/len(train_loader):.4f} | "
          f"val acc {val_acc:.3f} pr {val_p:.3f} rc {val_r:.3f} f1 {val_f1:.3f}")

# test final
te_acc, te_p, te_r, te_f1 = eval_loop(test_loader)
print(f"TEST acc {te_acc:.3f} pr {te_p:.3f} rc {te_r:.3f} f1 {te_f1:.3f}")


