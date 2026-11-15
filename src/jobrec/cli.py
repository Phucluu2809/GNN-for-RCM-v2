import argparse
import json
import sys
from pathlib import Path

from .baselines import evaluate_baseline
from .data import prepare, read_json, write_json
from .eda import write_eda, write_graph_svg
from .features import fit_preprocessors
from .graph import build_graph, training_context


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jobrec")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Audit CSVs, extract events, split by date, write artifacts and EDA")
    prepare_parser.add_argument("--config", default="configs/default.json")
    prepare_parser.add_argument("--data-dir")
    prepare_parser.add_argument("--output-dir")
    prepare_parser.add_argument("--jobs-file")
    compare_parser = commands.add_parser("compare", help="Paired bootstrap of per-user results on identical cohorts")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--metric", default="ndcg@20")
    compare_parser.add_argument("--output", default="artifacts/comparison.json")
    for name in ("graph", "baseline", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", default="artifacts/prepared")
        command.add_argument("--output-dir", default=f"artifacts/{name}")
        if name in {"graph", "train", "baseline"}:
            command.add_argument("--config", default="configs/default.json")
        if name in {"baseline", "train", "evaluate"}:
            command.add_argument("--max-users", type=int)
        if name in {"baseline", "evaluate"}:
            command.add_argument("--partition", choices=("validation", "test"), default="validation" if name == "baseline" else "test")
        if name == "baseline":
            command.add_argument("--model", choices=("random", "popular", "content", "logreg", "all"), default="all")
        if name == "train":
            command.add_argument("--model", choices=("sage", "rgcn", "hgt", "mf", "lightgcn"), required=True)
            command.add_argument("--gate", choices=("none", "uniform", "global", "profile", "history"), default="none")
            command.add_argument("--device", default="cpu")
            command.add_argument("--epochs", type=int)
            command.add_argument("--max-pairs", type=int)
        if name == "evaluate":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--device", default="cpu")
    arguments = parser.parse_args(argv)
    if hasattr(arguments, "max_users") and arguments.max_users is not None and arguments.max_users <= 0:
        parser.error("--max-users must be positive")
    try:
        if arguments.command == "compare":
            from .statistics import paired_bootstrap
            result = paired_bootstrap(read_json(arguments.left), read_json(arguments.right), arguments.metric)
            write_json(arguments.output, result)
            print(json.dumps(result, indent=2))
            return
        if arguments.command == "prepare":
            config = read_json(arguments.config)
            for name in ("data_dir", "output_dir", "jobs_file"):
                if getattr(arguments, name) is not None:
                    config[name] = getattr(arguments, name)
            bundle, audit, coverage = prepare(config)
            state = fit_preprocessors(bundle, config)
            write_json(Path(config["output_dir"]) / "preprocessors.json", state)
            count = write_eda(bundle, audit, Path(config["output_dir"]) / "eda")
            print(json.dumps({"output": config["output_dir"], "pairs": len(bundle["pairs"]),
                              "events": len(bundle["events"]), "cutoffs": [bundle["split"]["train_cutoff"], bundle["split"]["test_cutoff"]],
                              "coverage": coverage, "eda_charts": count}, indent=2))
            return
        bundle = read_json(Path(arguments.prepared) / "bundle.json.gz")
        output = Path(arguments.output_dir)
        if arguments.command == "graph":
            config = read_json(arguments.config)
            state = fit_preprocessors(bundle, config)
            graph = build_graph(training_context(bundle), state, config.get("graph_level", "G2"),
                                edge_retention=config.get("edge_retention", 1), seed=config.get("seed", 42))
            write_json(output / "graph_summary.json", graph["summary"])
            write_json(output / "preprocessors.json", state)
            write_graph_svg(graph, output / "graph_schema.svg")
            print(json.dumps(graph["summary"], indent=2))
        elif arguments.command == "baseline":
            config = read_json(arguments.config)
            state = fit_preprocessors(bundle, config)
            names = ("random", "popular", "content") if arguments.model == "all" else (arguments.model,)
            for name in names:
                result, per_user, recommendations = evaluate_baseline(bundle, state, name, arguments.partition, config.get("seed", 42), arguments.max_users)
                destination = output / f"{name}_{arguments.partition}"
                write_json(destination / "metrics.json", result)
                write_json(destination / "per_user.json.gz", per_user)
                write_json(destination / "recommendations.json.gz", recommendations)
                print(json.dumps(result, indent=2))
        elif arguments.command == "train":
            from .training import train
            config = read_json(arguments.config)
            if arguments.epochs is not None:
                config["epochs"] = arguments.epochs
            profile = train(bundle, config, arguments.model, output, arguments.device, arguments.gate, arguments.max_users, arguments.max_pairs)
            print(json.dumps(profile, indent=2))
        else:
            from .training import evaluate_checkpoint
            result, per_user, recommendations = evaluate_checkpoint(bundle, arguments.checkpoint, arguments.partition, arguments.device, arguments.max_users)
            write_json(output / "metrics.json", result)
            write_json(output / "per_user.json.gz", per_user)
            write_json(output / "recommendations.json.gz", recommendations)
            print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"jobrec: {error}\n")
    except ModuleNotFoundError as error:
        parser.exit(1, f"jobrec: missing dependency {error.name}. Install the train extra for GNN commands: pip install -e \".[train]\"\n")


if __name__ == "__main__":
    main(sys.argv[1:])
