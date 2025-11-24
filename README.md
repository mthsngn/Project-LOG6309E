# Project-LOG6309E

**Log-based vs Graph-based Approaches to Fault Diagnosis**

## 📂 Repository Structure

```text
PROJECT-LOG6309E/
│
├── anomaly_detection/          # All anomaly detection models (baseline, BERT, GNN) for HDFS and BGL
│
├── bert_sequences/             # Scripts + generated BERT sequence CSVs
│   ├── *_BERT_seq.py           # Scripts that generate encoded sequences from logs
│   └── *.csv                   # Generated BERT sequences
│
├── datasets/                   # Raw datasets
│   ├── BGL/                    # BGL log datasets
│   └── tracebench/             # HDFS TraceBench dataset
│
├── fault_classification/       # Fault classification models (baseline, BERT, GNN) for HDFS
│
├── ipynb/                      # Jupyter notebooks for experiments and exploration
│
├── master_tables/              # Processed master tables
│   ├── BGL/                    # - events.csv, edges.csv, traces.csv, ...
│   └── HDFS/                   # - same structure for HDFS
│
├── preprocess_BGL/             # Preprocessing tools for BGL (downloader, parser, master tables generator)
│
├── preprocess_HDFS/            # Preprocessing tools for HDFS (master tables generator)
│
├── .gitignore
├── README.md
└── requirements.txt
