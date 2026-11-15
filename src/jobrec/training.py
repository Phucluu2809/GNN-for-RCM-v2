import platform
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as functional

from .data import NegativeSampler, digest, evaluation_context, write_json
from .eda import write_learning_curves
from .features import FEATURE_SCHEMA_VERSION, GROUPS, attributes, compatibility, fit_preprocessors
from .graph import build_graph, disjoint_edges, to_torch, training_context
from .metrics import aggregate_metrics
from .models import JobRecommender


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gate_inputs(graph, context, state, device):
    dimension = state["feature_dim"]
    history = torch.zeros((len(graph["nodes"]["candidate"]), dimension), device=device)
    counts = torch.zeros((history.shape[0], 1), device=device)
    job_features = torch.tensor(graph["features"]["job"], dtype=torch.float32, device=device).reshape(-1, dimension)
    for row in context["past"]:
        candidate_index = graph["mappings"]["candidate"].get(row["candidate_id"])
        job_index = graph["mappings"]["job"].get(row["job_id"])
        if candidate_index is not None and job_index is not None and row["time"] < context["cutoff"]:
            history[candidate_index] += job_features[job_index]
            counts[candidate_index] += 1
    history = history / counts.clamp_min(1)
    masks = []
    for identifier in graph["nodes"]["candidate"]:
        groups = attributes(context["candidates"][identifier], "candidate", state["use_llm"], state["salary_bounds"])
        masks.append([bool(groups[group]) and group != "salary" for group in GROUPS])
    return history, torch.tensor(masks, dtype=torch.bool, device=device)


def pair_features(candidate_ids, job_ids, context, state, device):
    values = [compatibility(context["candidates"][candidate_id], context["jobs"][job_id], state)[0]
              for candidate_id, job_id in zip(candidate_ids, job_ids)]
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(-1, 6)


@torch.no_grad()
def evaluate_model(model, graph, context, state, device, max_users=None, seed=42, chunk_size=1024):
    model.eval()
    model.set_node_ids(graph["nodes"])
    features, edges = to_torch(graph, device)
    embeddings = model(features, edges)
    history, masks = gate_inputs(graph, context, state, device) if model.gate is not None else (None, None)
    identifiers = graph["nodes"]["job"]
    users = sorted(context["positives"])
    if max_users is not None:
        users = random.Random(seed).sample(users, min(max_users, len(users)))
    recommendations = {}
    started = time.perf_counter()
    for candidate_id in users:
        ranked_scores = []
        available_jobs = [identifier for identifier in identifiers if identifier not in context["seen"].get(candidate_id, set())]
        for start in range(0, len(available_jobs), chunk_size):
            batch = available_jobs[start:start + chunk_size]
            candidate_indices = torch.full((len(batch),), graph["mappings"]["candidate"][candidate_id], device=device, dtype=torch.long)
            job_indices = torch.tensor([graph["mappings"]["job"][identifier] for identifier in batch], device=device)
            compatibilities = pair_features([candidate_id] * len(batch), batch, context, state, device) if model.gate is not None else None
            scores = model.score_pairs(embeddings, candidate_indices, job_indices, features["candidate"], history, masks, compatibilities)
            ranked_scores.extend(zip(batch, scores.cpu().tolist()))
        recommendations[candidate_id] = [identifier for identifier, _ in sorted(ranked_scores, key=lambda pair: (-pair[1], pair[0]))[:20]]
    positives = {identifier: context["positives"][identifier] for identifier in users}
    result, per_user = aggregate_metrics(recommendations, positives, len(identifiers), {row["candidate_id"] for row in context["past"]})
    result.update({"coverage": context["coverage"], "scoring_seconds": time.perf_counter() - started,
                   "evaluation": "full_catalog", "max_users": max_users})
    return result, per_user, recommendations


def train(bundle, config, architecture, output_dir, device="cpu", gate_mode="none", max_users=None, max_pairs=None):
    if config.get("epochs", 100) < 1 or config.get("negatives", 5) < 1 or config.get("patience", 10) < 1:
        raise ValueError("epochs, negatives and patience must be positive")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users must be positive")
    config = dict(config)
    if architecture in {"mf", "lightgcn"}:
        config["graph_level"] = "G0"
    seed = config.get("seed", 42)
    seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = fit_preprocessors(bundle, config)
    context = training_context(bundle)
    messages, supervision = disjoint_edges(context, config.get("supervision_fraction", 0.2), seed)
    graph = build_graph(context, state, config.get("graph_level", "G2"), messages, config.get("edge_retention", 1), seed)
    features, edges = to_torch(graph, device)
    metadata = (list(graph["nodes"]), [tuple(edge_type) for edge_type in graph["edge_types"]])
    arguments = {"architecture": architecture, "input_dim": state["feature_dim"], "hidden": config.get("hidden", 64),
                 "layers": config.get("layers", 2), "heads": config.get("heads", 4),
                 "dropout": config.get("dropout", 0.2), "gate_mode": gate_mode}
    if architecture in {"mf", "lightgcn"}:
        arguments["node_ids"] = {
            kind: sorted({row[f"{kind}_id"] for row in context["past"] if row[f"{kind}_id"] in graph["mappings"][kind]})
            for kind in ("candidate", "job")}
    model = JobRecommender(metadata=metadata, **arguments).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
    history_inputs, masks = gate_inputs(graph, {**context, "past": messages}, state, device) if model.gate is not None else (None, None)
    validation_context = evaluation_context(bundle, "validation")
    validation_graph = build_graph(validation_context, state, config.get("graph_level", "G2"))
    sampler, rng = NegativeSampler(context), random.Random(seed)
    train_log, best, stale = [], -1.0, 0
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, config.get("epochs", 100) + 1):
        model.train()
        model.set_node_ids(graph["nodes"])
        chosen = rng.sample(supervision, min(len(supervision), max_pairs)) if max_pairs else list(supervision)
        users, positive_jobs, negative_jobs = [], [], []
        for row in chosen:
            negatives = sampler.sample(row["candidate_id"], row["time"], rng, config.get("negatives", 5))
            users.extend([row["candidate_id"]] * len(negatives))
            positive_jobs.extend([row["job_id"]] * len(negatives))
            negative_jobs.extend(negatives)
        if not users:
            raise ValueError("No eligible negatives; inspect availability and interaction density")
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(features, edges)
        user_indices = torch.tensor([graph["mappings"]["candidate"][identifier] for identifier in users], device=device)
        positive_indices = torch.tensor([graph["mappings"]["job"][identifier] for identifier in positive_jobs], device=device)
        negative_indices = torch.tensor([graph["mappings"]["job"][identifier] for identifier in negative_jobs], device=device)
        positive_features = pair_features(users, positive_jobs, context, state, device) if model.gate is not None else None
        negative_features = pair_features(users, negative_jobs, context, state, device) if model.gate is not None else None
        positive_scores = model.score_pairs(embeddings, user_indices, positive_indices, features["candidate"], history_inputs, masks, positive_features)
        negative_scores = model.score_pairs(embeddings, user_indices, negative_indices, features["candidate"], history_inputs, masks, negative_features)
        loss = functional.softplus(negative_scores - positive_scores).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        result, _, _ = evaluate_model(model, validation_graph, validation_context, state, device, max_users, seed)
        if not result["users"]:
            raise ValueError("No eligible validation users")
        score = result["metrics"]["ndcg@20"]
        train_log.append({"epoch": epoch, "loss": loss.item(), "validation_ndcg20": score})
        print(f"{architecture} epoch={epoch} loss={loss.item():.5f} validation_ndcg20={score:.5f}", flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "model_arguments": arguments, "metadata": metadata,
                        "preprocessors": state, "config": config, "epoch": epoch,
                        "dataset_hash": bundle["manifest"]["dataset_hash"], "split_hash": bundle["split"]["sha256"],
                        "message_pairs": messages, "train_supervision_pairs": len(supervision)}, output / "best.pt")
        else:
            stale += 1
        write_json(output / "history.json", train_log)
        if stale >= config.get("patience", 10):
            break
    import torch_geometric
    profile = {"model": architecture, "gate_mode": gate_mode, "seed": seed,
               "dataset_hash": bundle["manifest"]["dataset_hash"], "split_hash": bundle["split"]["sha256"],
               "preprocessing_hash": digest(state), "feature_schema_version": state["feature_schema_version"],
               "train_seconds": time.perf_counter() - started, "best_validation_ndcg20": best,
               "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
               "python": platform.python_version(), "torch": torch.__version__, "pyg": torch_geometric.__version__,
               "device": str(device), "cuda": torch.version.cuda,
               "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2 if str(device).startswith("cuda") else None,
               "max_users": max_users, "max_pairs": max_pairs,
               "protocol": "fixed_disjoint_train_supervision; full_batch; validation_messages_all_past",
               "graph": graph["summary"]}
    write_json(output / "profile.json", profile)
    write_learning_curves(train_log, output)
    return profile


def evaluate_checkpoint(bundle, checkpoint_path, partition="test", device="cpu", max_users=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["preprocessors"].get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema changed; retrain this checkpoint")
    if checkpoint["dataset_hash"] != bundle["manifest"]["dataset_hash"] or checkpoint["split_hash"] != bundle["split"]["sha256"]:
        raise ValueError("Checkpoint dataset/split hash mismatch")
    model = JobRecommender(metadata=checkpoint["metadata"], **checkpoint["model_arguments"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    context = evaluation_context(bundle, partition)
    graph = build_graph(context, checkpoint["preprocessors"], checkpoint["config"].get("graph_level", "G2"))
    result, per_user, recommendations = evaluate_model(model, graph, context, checkpoint["preprocessors"], device, max_users, checkpoint["config"].get("seed", 42))
    result.update({"partition": partition, "model": checkpoint["model_arguments"]["architecture"],
                   "preprocessing_hash": digest(checkpoint["preprocessors"]),
                   "feature_schema_version": checkpoint["preprocessors"]["feature_schema_version"],
                   "gate_mode": checkpoint["model_arguments"]["gate_mode"], "checkpoint_epoch": checkpoint["epoch"],
                   "dataset_hash": checkpoint["dataset_hash"], "split_hash": checkpoint["split_hash"],
                   "context_policy": "frozen_at_partition_start; test_uses_past_validation_events_without_refit"})
    return result, per_user, recommendations
