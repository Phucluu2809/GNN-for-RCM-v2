import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jobrec.data import read_json, write_json
from jobrec.training import evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description="Validation-only multi-seed benchmark; does not automatically evaluate test")
    parser.add_argument("--prepared", default="artifacts/prepared")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--models", nargs="+", choices=("sage", "rgcn", "hgt", "mf", "lightgcn"), default=["sage", "rgcn", "hgt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--gate", choices=("none", "uniform", "global", "profile", "history"), default="none")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-pairs", type=int)
    arguments = parser.parse_args()
    bundle = read_json(Path(arguments.prepared) / "bundle.json.gz")
    base_config = read_json(arguments.config)
    summary = []
    for architecture in arguments.models:
        for seed in arguments.seeds:
            output = Path(arguments.output_dir) / f"{architecture}_{arguments.gate}_{seed}"
            if (output / "best.pt").exists():
                raise FileExistsError(f"Refusing to overwrite an existing experiment: {output}")
            config = {**base_config, "seed": seed}
            profile = train(bundle, config, architecture, output, arguments.device, arguments.gate, arguments.max_users, arguments.max_pairs)
            result, per_user, _ = evaluate_checkpoint(bundle, output / "best.pt", "validation", arguments.device, arguments.max_users)
            write_json(output / "validation_metrics.json", result)
            write_json(output / "validation_per_user.json.gz", per_user)
            summary.append({"run": str(output), "seed": seed, "model": architecture, "gate": arguments.gate,
                            "validation_metrics": result["metrics"], "profile": profile})
            write_json(Path(arguments.output_dir) / "summary.json", summary)


if __name__ == "__main__":
    main()
