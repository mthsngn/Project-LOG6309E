import os
import requests

DATASETS = {
    "BGL": "https://zenodo.org/records/8196385/files/BGL.zip"
}

os.makedirs("../datasets", exist_ok=True)

def download_file(url, dest):
    print(f"Téléchargement de {dest} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Terminé : {dest}")

for name, url in DATASETS.items():
    dest_path = os.path.join("../datasets", f"{name}.tar.gz")
    download_file(url, dest_path)
