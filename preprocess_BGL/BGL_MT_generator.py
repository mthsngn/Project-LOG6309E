import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# --- Configuration des chemins ---
DATA_DIR = "datasets"
PARSED_DIR = "parsed_logs"
STRUCTURED_DIR = "../master_tables/BGL/test"
os.makedirs(STRUCTURED_DIR, exist_ok=True)

# --- Utility function: find file ---
def find_file(base_dir, extension):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(extension):
                return os.path.join(root, f)
    return None

# ============================================================
# === BGL Trace Construction (NodeID + Windows)
# ============================================================

WINDOW_SIZE = 50
STRIDE = 50

def build_bgl_traces(df):
    print("-> Construction des traces BGL par NodeID...")

    df = df.sort_values(["Node", "Timestamp"])
    traces = []
    events = []
    edges = []

    task_id = 0

    for node, node_df in tqdm(df.groupby("Node"), desc="   -> Groupes NodeID"):
        node_df = node_df.sort_values("Timestamp").reset_index(drop=True)

        labels = node_df["Label"].values
        length = len(node_df)

        # Sliding windows over this node
        for start in range(0, length - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            window = node_df.iloc[start:end]
            window_labels = labels[start:end]

            is_abnormal = int(any(lbl != '-' for lbl in window_labels))

            # --- traces.csv---
            traces.append({
                "TaskID": task_id,
                "Node": node,
                "start_timestamp": int(window["Timestamp"].iloc[0]),
                "end_timestamp": int(window["Timestamp"].iloc[-1]),
                "IsAbnormal": is_abnormal,
                "num_events": len(window)
            })

            # --- events.csv---
            for tid, (_, row) in enumerate(window.iterrows()):
                events.append({
                    "TaskID": task_id,
                    "TID": tid,
                    "LineId": row.get("LineId", None),
                    "Timestamp": row["Timestamp"],
                    "Node": node,
                    "EventId": row["EventId"],
                    "EventTemplate": row["EventTemplate"],
                    "Label": row["Label"],
                    "Component": row.get("Component", None),
                    "Level": row.get("Level", None),
                    "Content": row.get("Content", None),
                })

            # --- edges.csv---
            for tid in range(len(window) - 1):
                edges.append({
                    "TaskID": task_id,
                    "FatherTID": tid,
                    "ChildTID": tid + 1
                })

            task_id += 1

    # Convert to DataFrames
    traces_df = pd.DataFrame(traces)
    events_df = pd.DataFrame(events)
    edges_df  = pd.DataFrame(edges)

    # Save output
    traces_df.to_csv(os.path.join(STRUCTURED_DIR, "traces.csv"), index=False)
    events_df.to_csv(os.path.join(STRUCTURED_DIR, "events.csv"), index=False)
    edges_df.to_csv(os.path.join(STRUCTURED_DIR, "edges.csv"), index=False)

    print(f"   ->✅ Sauvegardé : {len(traces_df)} traces, {len(events_df)} events, {len(edges_df)} edges.")

def structure_and_label_data(dataset_name, parsed_csv_path, base_dir):
    print(f"\n--- Étape 2: Structuration de {dataset_name} ---")
    df = pd.read_csv(parsed_csv_path)

    if dataset_name == "HDFS":
        print("   -> ℹ️  Construction de traces HDFS (BlockID).")
        print("   -> ✅ Structuration HDFS terminée (traces.csv, events.csv, edges.csv créés).")

    elif dataset_name == "BGL":
        print("   -> ℹ️  Construction de traces BGL (NodeID + fenêtres).")
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        df['Timestamp'] = df['Timestamp'].astype(int)
        build_bgl_traces(df)

        print("   -> ✅ Structuration BGL terminée (traces.csv, events.csv, edges.csv créés).")


# --- Script Principal ---
if __name__ == "__main__":
    for dataset in ["BGL"]:
        base_dir = os.path.join(DATA_DIR, dataset)
        parsed_csv_path = os.path.join(PARSED_DIR, f"{dataset}_parsed.csv")
        structure_and_label_data(dataset, parsed_csv_path, base_dir)

    print("\n🎉 Tous les datasets ont été traités.")