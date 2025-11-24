import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from drain3.masking import MaskingInstruction

# --- Configuration des chemins ---
DATA_DIR = "../datasets"
PARSED_DIR = "parsed_logs"
STATE_ROOT_DIR = "drain_states"
os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(STATE_ROOT_DIR, exist_ok=True)
OPEN_KW = dict(encoding="utf-8", errors="replace")

# --- Regex HDFS ---
HDFS_LOG_FORMAT_RE = re.compile(
    r"^(?P<Date>\d{6})\s+(?P<Time>\d{6})\s+(?P<PID>\d+)\s+(?P<Level>\w+)\s+(?P<Component>[^:]+):\s+(?P<Content>.*)$")
HDFS_BLOCK_ID_RE = re.compile(r"(blk_-?\d+)")


# --- Fonctions Utilitaires ---
def find_file(base_dir, extension):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(extension):
                return os.path.join(root, f)
    return None


def extract_parameters_before_masking(content, template, masking_rules=None):
    """Extrait les paramètres en appliquant les mêmes règles de masquage que Drain3."""
    if not template or template == content:
        return []

    # Appliquer les règles de masquage pour identifier les zones à extraire
    params = []

    if masking_rules:
        for rule in masking_rules:
            # MaskingInstruction stocke le regex dans 'regex' (regex compilé)
            regex = rule.regex if hasattr(rule, 'regex') else re.compile(rule.pattern)
            matches = list(regex.finditer(content))
            for match in matches:
                params.append(match.group())

    # Si pas de masking rules, utiliser la méthode par pattern
    if not params:
        pattern = re.escape(template).replace(r'\<\*\>', '(.*?)')
        pattern = '^' + pattern + '$'
        match = re.match(pattern, content)
        if match:
            params = [p.strip() for p in match.groups()]

    return params



def extract_parameters(content, template):
    """Extrait les paramètres en comparant le contenu au template."""
    if not template or template == content:
        return []

    # Échapper les caractères spéciaux du template sauf <*>
    pattern = re.escape(template).replace(r'\<\*\>', '(.*?)')

    # Ajouter des ancres pour matcher exactement
    pattern = '^' + pattern + '$'

    match = re.match(pattern, content)
    if match:
        # Nettoyer les paramètres (enlever espaces superflus)
        return [p.strip() for p in match.groups()]
    return []


def setup_drain(dataset_name):
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    persistence = FilePersistence(os.path.join(STATE_ROOT_DIR, f"{dataset_name}_drain_state.bin"))

    config.masking_instructions = [
        MaskingInstruction(re.compile(r'blk_-?\d+'), "<*>"),
        MaskingInstruction(re.compile(r'(\d+\.){3}\d+:\d+'), "<*>")
    ]
    return TemplateMiner(persistence, config)


# --- Fonctions de Parsing ---
def parse_hdfs_advanced(log_path, miner):
    parsed_lines = []
    masking_rules = miner.config.masking_instructions if hasattr(miner.config, 'masking_instructions') else None

    with open(log_path, 'r', **OPEN_KW) as f:
        for line_id, line_content in tqdm(enumerate(f, 1), desc="🧠  Parsing HDFS"):
            match = HDFS_LOG_FORMAT_RE.match(line_content.strip())
            if not match:
                continue

            log_data = match.groupdict()
            block_ids = HDFS_BLOCK_ID_RE.findall(log_data['Content'])
            if not block_ids:
                continue

            result = miner.add_log_message(log_data['Content'])

            params = extract_parameters_before_masking(
                log_data['Content'],
                result['template_mined'],
                masking_rules
            )

            log_data.update({
                "LineId": line_id, "BlockId": block_ids[0],
                "EventTemplate": result['template_mined'],
                "EventId": result['cluster_id'],
                "ParameterList": str(params)
            })
            parsed_lines.append(log_data)
    return pd.DataFrame(parsed_lines)


def parse_log_file(dataset_name, log_path):
    print(f"\n--- Étape 1: Parsing de {dataset_name} ---")
    miner = setup_drain(dataset_name)
    df = parse_hdfs_advanced(log_path, miner)
    out_path = os.path.join(PARSED_DIR, f"{dataset_name}_parsed.csv")
    df.to_csv(out_path, index=False)
    print(f"   -> ✅ Parsing terminé : {out_path}")
    return out_path

# --- Script Principal ---
if __name__ == "__main__":
    for dataset in ["HDFS"]:
        base_dir = os.path.join(DATA_DIR, dataset)
        log_file_path = find_file(base_dir, ".log")
        if not log_file_path:
            print(f"Fichier .log non trouvé pour {dataset}, dataset ignoré.")
            continue
        parse_log_file(dataset, log_file_path)
    print("\n🎉 Tous les datasets ont été parsés.")
