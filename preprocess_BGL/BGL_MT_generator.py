import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# --- Configuration des chemins ---
DATA_DIR = "../datasets"
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
# === BGL Trace Construction (Windows)
# ============================================================

# 6-hour window in seconds
SESSION_SECONDS = 6 * 3600

def build_bgl_traces(df, structured_dir=STRUCTURED_DIR):
    print("-> Construction des traces BGL par fenêtres temporelles de 6h...")

    df = df.copy()
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Timestamp"] = df["Timestamp"].astype(int)

    # Définir les fenêtres de 6h (SessionId)
    df["SessionId"] = df["Timestamp"] // SESSION_SECONDS

    # Trier globalement par SessionId puis Timestamp
    df = df.sort_values(["SessionId", "Timestamp"]).reset_index(drop=True)

    traces = []
    events = []
    edges = []

    task_id = 0

    # Groupement par fenêtre temporelle (SessionId)
    for session_id, session_df in tqdm(df.groupby("SessionId"), desc="   -> Fenêtres 6h"):
        window = session_df.sort_values("Timestamp").reset_index(drop=True)

        # Déterminer si la fenêtre est anormale
        labels = window["Label"].values
        is_abnormal = int(any(lbl != "-" for lbl in labels))

        # --- traces.csv ---
        traces.append({
            "TaskID": task_id,
            "SessionId": int(session_id),
            "start_timestamp": int(window["Timestamp"].iloc[0]),
            "end_timestamp": int(window["Timestamp"].iloc[-1]),
            "IsAbnormal": is_abnormal,
            "num_events": len(window),
        })

        # --- events.csv ---
        for tid, (_, row) in enumerate(window.iterrows()):
            events.append({
                "TaskID": task_id,
                "TID": tid,
                "LineId": row.get("LineId", None),
                "Timestamp": int(row["Timestamp"]),
                "Node": row.get("Node", None),
                "Date": row.get("Date", None),
                "Time": row.get("Time", None),
                "NodeRepeat": row.get("NodeRepeat", None),
                "Type": row.get("Type", None),
                "Component": row.get("Component", None),
                "Level": row.get("Level", None),
                "Content": row.get("Content", None),
                "EventId": row.get("EventId", None),
                "EventTemplate": row.get("EventTemplate", None),
                "ParameterList": row.get("ParameterList", None),
                "Label": row.get("Label", None),
            })

        # --- edges.csv ---
        for tid in range(len(window) - 1):
            edges.append({
                "TaskID": task_id,
                "FatherTID": tid,
                "ChildTID": tid + 1,
            })

        task_id += 1

    # Conversion en DataFrames
    traces_df = pd.DataFrame(traces)
    events_df = pd.DataFrame(events)
    edges_df  = pd.DataFrame(edges)

    # Sauvegarde
    os.makedirs(structured_dir, exist_ok=True)
    traces_df.to_csv(os.path.join(structured_dir, "traces.csv"), index=False)
    events_df.to_csv(os.path.join(structured_dir, "events.csv"), index=False)
    edges_df.to_csv(os.path.join(structured_dir, "edges.csv"), index=False)

    print(f"   -> ✅ Sauvegardé : {len(traces_df)} traces, {len(events_df)} events, {len(edges_df)} edges.")

def structure_and_label_data(dataset_name, parsed_csv_path, base_dir):
    print(f"\nStructuration de {dataset_name} ---")
    df = pd.read_csv(parsed_csv_path)
    print("   -> ℹ️  Construction de traces BGL")
    build_bgl_traces(df)
    print("   -> ✅ Structuration BGL terminée (traces.csv, events.csv, edges.csv créés).")


# --- Script Principal ---
if __name__ == "__main__":
    for dataset in ["BGL"]:
        base_dir = os.path.join(DATA_DIR, dataset)
        parsed_csv_path = os.path.join(PARSED_DIR, f"{dataset}_parsed.csv")
        structure_and_label_data(dataset, parsed_csv_path, base_dir)

    print("\n🎉 Tous les datasets ont été traités.")