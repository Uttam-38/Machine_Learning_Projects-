import zipfile
import requests
from pathlib import Path

from src.config import DataConfig, ensure_dirs

def download_file(url: str, dest: Path):
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

def main():
    paths = ensure_dirs()
    cfg = DataConfig()

    zip_path = paths.raw / cfg.zip_name
    extract_dir = paths.raw / cfg.folder_name

    if extract_dir.exists():
        print(f"[OK] Already extracted: {extract_dir}")
        return

    if not zip_path.exists():
        print(f"[DL] Downloading MovieLens from {cfg.movielens_url}")
        download_file(cfg.movielens_url, zip_path)
        print(f"[OK] Downloaded: {zip_path}")

    print("[EXTRACT] Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(paths.raw)
    print(f"[OK] Extracted to: {paths.raw}")

if __name__ == "__main__":
    main()
