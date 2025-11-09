import os
import pandas as pd

CSV_PATH = "master_tables"


def build_encoder_sequences(events_df, traces_df, join_cols=("OpName","Description"), order_cols=("StartTime","TID")):
    # Act on event_df
    df = events_df.copy()

    # Sort by TaskID then StartTime then TID (Order events in Trace)
    if order_cols: df = df.sort_values(["TaskID"] + list(order_cols))

    # Create events text column = content of join_cols
    def get_events_text(row): return " : ".join(str(row[c]) for c in join_cols if c in row and pd.notna(row[c]))
    df["EventText"] = df.apply(get_events_text, axis=1)

    # Group by TaskID and aggregate text
    seq = df.groupby("TaskID")["EventText"].apply(lambda s: " ".join([t for t in s if t])).reset_index()

    # Merge labels and info from traces_df
    labels = traces_df[["TaskID","IsAbnormal","FaultType","Category"]]
    return seq.merge(labels, on="TaskID", how="left")

if __name__ == "__main__":
    traces_df = pd.read_csv(os.path.join(CSV_PATH, "traces.csv"))
    events_df = pd.read_csv(os.path.join(CSV_PATH, "events.csv"))
    edges_df = pd.read_csv(os.path.join(CSV_PATH, "edges.csv"))
    ops_df = pd.read_csv(os.path.join(CSV_PATH, "operations.csv"))

    sequences = build_encoder_sequences(events_df, traces_df)
    print(sequences.head())