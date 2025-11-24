import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# --- Configuration des chemins ---
DATA_DIR = "../datasets"
PARSED_DIR = "parsed_logs"
STRUCTURED_DIR = "../master_tables/HDFS/test"
os.makedirs(STRUCTURED_DIR, exist_ok=True)

# --- Utility function: find file ---
def find_file(base_dir, extension):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(extension):
                return os.path.join(root, f)
    return None

# ============================================================
# === HDFS Trace Construction (Block)
# ============================================================

def build_hdfs_traces(df, label_df, structured_dir=STRUCTURED_DIR):
    print("-> Construction des traces HDFS par BlockId...")

    df = df.copy()

    # Fusionner les labels sur BlockId
    merged = pd.merge(df, label_df, on="BlockId", how="left")
    merged["Label"] = merged["Label"].fillna("Normal")

    # LineId pour l'ordre dans chaque bloc
    if "LineId" in merged.columns:
        merged["LineId"] = pd.to_numeric(merged["LineId"], errors="coerce")
        merged = merged.dropna(subset=["LineId"])
        merged["LineId"] = merged["LineId"].astype(int)
        merged = merged.sort_values(["BlockId", "LineId"]).reset_index(drop=True)
    else:
        # fallback : ordre du fichier
        merged = merged.reset_index(drop=True)
        merged["LineId"] = merged.index
        merged = merged.sort_values(["BlockId", "LineId"]).reset_index(drop=True)

    traces = []
    events = []
    edges = []

    task_id = 0

    # Groupement par BlockId
    for block_id, block_df in tqdm(merged.groupby("BlockId"), desc="   -> Groupes BlockId"):
        window = block_df.reset_index(drop=True)

        # Trace anormale si au moins un event Label != 'Normal'
        is_abnormal = int(any(lbl != "Normal" for lbl in window["Label"].values))

        # ----- traces.csv -----
        traces.append({
            "TaskID": task_id,
            "BlockId": block_id,
            "IsAbnormal": is_abnormal,
            "num_events": len(window),
        })

        # ----- events.csv -----
        for tid, (_, row) in enumerate(window.iterrows()):
            events.append({
                "TaskID": task_id,
                "TID": tid,
                "Date": row.get("Date", None),
                "Time": row.get("Time", None),
                "PID": row.get("PID", None),
                "Level": row.get("Level", None),
                "Component": row.get("Component", None),
                "Content": row.get("Content", None),
                "LineId": row.get("LineId", None),
                "BlockId": row.get("BlockId", None),
                "EventTemplate": row.get("EventTemplate", None),
                "EventId": row.get("EventId", None),
                "ParameterList": row.get("ParameterList", None),
                "Label": row.get("Label", None),  # Normal / Anomaly
            })

        # ----- edges.csv -----
        for tid in range(len(window) - 1):
            edges.append({
                "TaskID": task_id,
                "FatherTID": tid,
                "ChildTID": tid + 1,
            })

        task_id += 1

    # Conversion et sauvegarde
    traces_df = pd.DataFrame(traces)
    events_df = pd.DataFrame(events)
    edges_df  = pd.DataFrame(edges)

    os.makedirs(structured_dir, exist_ok=True)
    traces_df.to_csv(os.path.join(structured_dir, "traces.csv"), index=False)
    events_df.to_csv(os.path.join(structured_dir, "events.csv"), index=False)
    edges_df.to_csv(os.path.join(structured_dir, "edges.csv"), index=False)

    print(f"   -> ✅ HDFS tracé : {len(traces_df)} traces, {len(events_df)} events, {len(edges_df)} edges.")

def structure_and_label_data(dataset_name, parsed_csv_path, base_dir):
    print(f"\nStructuration de {dataset_name} ---")
    df = pd.read_csv(parsed_csv_path)
    label_df = pd.read_csv(find_file(base_dir, "anomaly_label.csv"))
    print("   -> ℹ️  Construction de traces HDFS (BlockID).")
    build_hdfs_traces(df, label_df)
    print("   -> ✅ Structuration HDFS terminée (traces.csv, events.csv, edges.csv créés).")


# --- Script Principal ---
if __name__ == "__main__":
    for dataset in ["HDFS"]:
        base_dir = os.path.join(DATA_DIR, dataset)
        parsed_csv_path = os.path.join(PARSED_DIR, f"{dataset}_parsed.csv")
        structure_and_label_data(dataset, parsed_csv_path, base_dir)
    print("\n🎉 Tous les datasets ont été traités.")