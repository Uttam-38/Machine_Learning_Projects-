import argparse
import subprocess
from itertools import product

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    methods = ["erm", "iw", "coral", "groupdro"]
    shift_types = ["none", "region", "device", "region_device", "concept"]
    severities = [0.0, 0.4, 0.7, 0.9]

    for m, st, sev in product(methods, shift_types, severities):
        cmd = [
            "python", "scripts/run_experiment.py",
            "--config", args.config,
            "--method", m,
        ]
        print("\n>>>", " ".join(cmd), f"(shift={st}, severity={sev})")
        # override via env vars (simple + CI-friendly)
        env = dict(**__import__("os").environ)
        env["SHIFT_TYPE"] = st
        env["SHIFT_SEVERITY"] = str(sev)
        subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
