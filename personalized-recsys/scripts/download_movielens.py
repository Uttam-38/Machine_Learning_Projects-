import os
import zipfile
import urllib.request
import shutil

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"  # stable host
ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_file(url: str, dest_path: str) -> None:
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as resp, open(dest_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"Saved to: {dest_path}")

def extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"Extracted to: {extract_to}")

def main(dataset: str = "1m"):
    os.makedirs("data/raw", exist_ok=True)

    if dataset.lower() == "1m":
        url = ML_1M_URL
        zip_name = "ml-1m.zip"
        folder_name = "ml-1m"
    elif dataset.lower() in ["100k", "100k.zip"]:
        url = ML_100K_URL
        zip_name = "ml-100k.zip"
        folder_name = "ml-100k"
    else:
        raise ValueError("dataset must be '1m' or '100k'")

    zip_path = os.path.join("data/raw", zip_name)

    # download
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    else:
        print(f"Zip already exists: {zip_path}")

    # extract
    extract_zip(zip_path, "data/raw")

    # move expected files into data/raw root so our loader works directly
    extracted_dir = os.path.join("data/raw", folder_name)
    if os.path.isdir(extracted_dir):
        print(f"Found extracted folder: {extracted_dir}")

        # For 1M, we need ratings.dat and movies.dat at data/raw/
        if folder_name == "ml-1m":
            for fname in ["ratings.dat", "movies.dat"]:
                src = os.path.join(extracted_dir, fname)
                dst = os.path.join("data/raw", fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"Copied {fname} -> data/raw/")
        # For 100K, we need u.data and u.item at data/raw/
        else:
            for fname in ["u.data", "u.item"]:
                src = os.path.join(extracted_dir, fname)
                dst = os.path.join("data/raw", fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"Copied {fname} -> data/raw/")

    print("\n✅ Done. Check data/raw/ now contains:")
    print("  - ratings.dat & movies.dat (for 1M) OR")
    print("  - u.data & u.item (for 100K)")

if __name__ == "__main__":
    # change to "100k" if you want the smaller dataset
    main(dataset="1m")
