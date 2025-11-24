import os
import pandas as pd

CSV_PATH = "../master_tables/BGL/test"

def build_encoder_sequences(events_df, traces_df, join_cols=("EventTemplate",), order_cols=("Timestamp", "TID")):
    df = events_df.copy()

    # Sort by TaskID then Timestamp then TID (Order events in Trace)
    if order_cols:
        df = df.sort_values(["TaskID"] + list(order_cols))

    # Create events text column = content of join_cols
    def get_events_text(row):
        return " : ".join(
            str(row[c]) for c in join_cols
            if c in row and pd.notna(row[c])
        )

    df["EventText"] = df.apply(get_events_text, axis=1)

    # Group by TaskID and aggregate text
    seq = (
        df.groupby("TaskID")["EventText"]
          .apply(lambda s: " ".join([t for t in s if t]))
          .reset_index()
    )

    # Merge labels and info from traces_df
    labels = traces_df[["TaskID", "IsAbnormal"]]
    return seq.merge(labels, on="TaskID", how="left")

if __name__ == "__main__":
    traces_df = pd.read_csv(os.path.join(CSV_PATH, "traces.csv"))
    events_df = pd.read_csv(os.path.join(CSV_PATH, "events.csv"))
    edges_df = pd.read_csv(os.path.join(CSV_PATH, "edges.csv"))

    encoder_seq = build_encoder_sequences(events_df, traces_df)
    encoder_seq.to_csv('BGL_encoder_seq.csv', index=False) 