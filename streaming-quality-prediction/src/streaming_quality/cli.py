import argparse
import subprocess

def main():
    ap = argparse.ArgumentParser(prog="sq")
    ap.add_argument("cmd", choices=["gen", "feat", "train", "eval", "api"])
    args = ap.parse_args()

    commands = {
        "gen":  ["python", "-m", "streaming_quality.data.generate_synthetic", "--out", "data/processed/sessions.parquet", "--n", "200000"],
        "feat": ["python", "-m", "streaming_quality.features.build_features", "--in", "data/processed/sessions.parquet", "--out", "data/processed/features.parquet"],
        "train":["python", "-m", "streaming_quality.models.train", "--data", "data/processed/features.parquet"],
        "eval": ["python", "-m", "streaming_quality.models.evaluate", "--data", "data/processed/features.parquet"],
        "api":  ["uvicorn", "streaming_quality.api.app:app", "--reload", "--port", "8000"],
    }
    subprocess.check_call(commands[args.cmd])

if __name__ == "__main__":
    main()
