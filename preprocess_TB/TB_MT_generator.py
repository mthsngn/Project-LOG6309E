import os
import re
from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd

DATA_PATH = "../datasets/tracebench_test"
CSV_PATH = "../master_tables/TB/test"

def fixed_read_csv(path):
    """Fix event.csv dynamically by quoting the Description field."""
    fixed_lines = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        fixed_lines.append(header)
        for line in f:
            line = line.rstrip('\n')
            # Split only the first 8 commas: 9 columns total
            parts = line.split(',', 8)
            if len(parts) == 9:
                # If the last field (Description) contains commas but is unquoted, wrap it
                desc = parts[-1]
                if not desc.startswith('"') and ',' in desc:
                    parts[-1] = f'"{desc}"'
                fixed_lines.append(','.join(parts))

    # Now read the corrected lines with pandas
    from io import StringIO
    buffer = StringIO('\n'.join(fixed_lines))
    df = pd.read_csv(buffer)
    return df

@dataclass
class TraceSet:
    name: str
    dirpath: str
    labels: Dict[str, Optional[str]]
    trace: pd.DataFrame
    event: pd.DataFrame
    edge: pd.DataFrame
    operation: pd.DataFrame

def extract_structure_from_folder(dirname):
    """Extract metadata from folder name."""
    _ABN_PREFIX = re.compile(r"^(AN|NM)_", re.IGNORECASE)
    d = dirname.strip('/')
    m = _ABN_PREFIX.match(d)
    if not m:
        raise ValueError(f"Invalid folder name format: {dirname}")
    
    # Label
    is_abn = (m.group(1).upper() == 'AN')
    parts = d.split('_')

    # Category + Fault Type
    category = None
    fault = None

    if is_abn:
        if len(parts) >= 3:
            category = parts[1]
            fault = parts[2]
        start_idx = 3
    else:
        if len(parts) >= 2:
            category = parts[1]
        start_idx = 2

    
    # Workload + Variables
    workload = None
    variables = None

    allowed_workloads = {'r', 'w', 'rw', 'rpc', 'rwrpc'}
    for i in range(start_idx, len(parts)):
        tok = parts[i].lower()
        if tok in allowed_workloads:
            workload = tok
            if i + 1 < len(parts):
                # Keep the rest
                variables = '_'.join(parts[i+1:])
            break

    return {
        'is_abnormal': int(is_abn),
        'class_label': 'Abnormal' if is_abn else 'Normal',
        'category': category,
        'fault_type': fault,
        'workload': workload,
        'variables': variables,
    }


def load_trace_set(dirpath):
    # Extract necessary metadata from folder name
    name = os.path.basename(dirpath)
    labels = extract_structure_from_folder(name)

    try:
        print(f"Loading: {name}")
        trace = pd.read_csv(os.path.join(dirpath, "trace.csv"))
        event = fixed_read_csv(os.path.join(dirpath, "event.csv"))
        edge = pd.read_csv(os.path.join(dirpath, "edge.csv"))
        operation = pd.read_csv(os.path.join(dirpath, "operation.csv"))

        return TraceSet(name, dirpath, labels, trace, event, edge, operation)

    except Exception as e:
        print(f"Error reading files in folder: {dirpath}")
        print(f"Error message: {type(e).__name__}: {e}")
        raise

def build_master_tables(sets):
    traces, events, edges, ops = [], [], [], []

    for ts in sets:
        labels = ts.labels or {}
        meta = {
            # "SetName":    ts.name,
            "Label": labels.get("class_label"),
            "IsAbnormal": labels.get("is_abnormal"),
            "FaultType":  labels.get("fault_type"),
            "Category":    labels.get("category"),
            "Workload":    labels.get("workload"),
            "Variables":   labels.get("variables"),
        }
        traces.append(ts.trace.assign(**meta))
        events.append(ts.event)
        edges.append(ts.edge)
        ops.append(ts.operation)

    traces_df = pd.concat(traces, ignore_index=True)
    events_df = pd.concat(events, ignore_index=True)
    edges_df  = pd.concat(edges,  ignore_index=True)
    ops_df    = pd.concat(ops,    ignore_index=True)
    
    return traces_df, events_df, edges_df, ops_df

def load_tracebench(root_dir):
    # Load all trace sets
    sets = [
        load_trace_set(os.path.join(root_dir, name))
        for name in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    print(f"Loaded {len(sets)} trace sets from {root_dir}")

    # Build master tables
    traces_df, events_df, edges_df, ops_df = build_master_tables(sets)
    print(f"Loaded {len(sets)} sets from {root_dir}")
    print(f"traces_df:   {traces_df.shape}")
    print(f"events_df:   {events_df.shape}")
    print(f"edges_df:    {edges_df.shape}")
    print(f"ops_df:      {ops_df.shape}")
    
    return traces_df, events_df, edges_df, ops_df

if __name__ == "__main__":
    traces_df, events_df, edges_df, ops_df = load_tracebench(DATA_PATH)
    
    # Save master tables
    os.makedirs(CSV_PATH, exist_ok=True)

    # Save master tables
    traces_df.to_csv(os.path.join(CSV_PATH, "traces.csv"), index=False)
    events_df.to_csv(os.path.join(CSV_PATH, "events.csv"), index=False)
    edges_df.to_csv(os.path.join(CSV_PATH, "edges.csv"), index=False)
    ops_df.to_csv(os.path.join(CSV_PATH, "operations.csv"), index=False)
    print(f"Saved master tables to: {CSV_PATH}")
