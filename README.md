# Project-LOG6309E

**Log-based vs Graph-based Approaches to Fault Diagnosis**

## 📂 Repository Structure

```text
PROJECT-LOG6309E/
│
├── anomaly_detection/          # All anomaly detection models (baseline, BERT, GNN) for TraceBench, BGL and HDFS
│
├── bert_sequences/             # Scripts + generated BERT sequence CSVs
│   ├── *_BERT_seq.py           # Scripts that generate encoded sequences from logs
│   └── *.csv                   # Generated BERT sequences
│
├── datasets/                   # Raw datasets
│   ├── tracebench/             # TraceBench dataset
│   ├── BGL/                    # BGL dataset (.log)
│   └── HDFS/                   # HDFS dataset (.log + anomaly.csv)
│
├── fault_classification/       # Fault classification models (baseline, BERT, GNN) for Tracebench
│
├── ipynb/                      # Jupyter notebooks for experiments and exploration
│
├── master_tables/              # Processed master tables
│   ├── TB/                     # - events.csv, edges.csv, traces.csv, ...
│   ├── BGL/                    # - same structure for BGL
│   └── HDFS/                   # - same structure for HDFS
│
├── preprocess_TB/              # Preprocessing tools for TraceBench (master tables generator)
│
├── preprocess_BGL/             # Preprocessing tools for BGL (downloader, parser, master tables generator)
│
├── preprocess_HDFS/            # Preprocessing tools for HDFS (downloader, parser, master tables generator)
│
├── .gitignore
├── README.md
└── requirements.txt
