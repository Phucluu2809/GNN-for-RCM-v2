import random
from collections import defaultdict

from .data import available_profiles, digest
from .features import GROUPS, attributes, hashed_vector, profile_vector


def training_context(bundle):
    cutoff = bundle["split"]["train_cutoff"]
    candidates, _ = available_profiles(bundle["candidates"], "candidate_id", cutoff)
    jobs, _ = available_profiles(bundle["jobs"], "job_id", cutoff)
    return {"cutoff": cutoff, "candidates": candidates, "jobs": jobs,
            "past": bundle["split"]["partitions"]["train"]}


def disjoint_edges(context, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("supervision_fraction must be strictly between 0 and 1")
    eligible = [row for row in context["past"]
                if row["candidate_id"] in context["candidates"] and row["job_id"] in context["jobs"]]
    if len(eligible) < 2:
        raise ValueError("Need at least two available training pairs")
    eligible = sorted(eligible, key=lambda row: (row["candidate_id"], row["job_id"]))
    random.Random(seed).shuffle(eligible)
    count = min(len(eligible) - 1, max(1, round(len(eligible) * fraction)))
    return eligible[count:], eligible[:count]


def build_graph(context, state, graph_level="G2", message_pairs=None, edge_retention=1.0, seed=42):
    if graph_level not in {"G0", "G1", "G2"}:
        raise ValueError("graph_level must be G0, G1 or G2")
    if not 0 < edge_retention <= 1:
        raise ValueError("edge_retention must be in (0, 1]")
    groups = [] if graph_level == "G0" else list(GROUPS[:2] if graph_level == "G1" else GROUPS)
    groups = [group for group in groups if state["vocab"][group]]
    nodes = {"candidate": sorted(context["candidates"]), "job": sorted(context["jobs"])}
    nodes.update({group: list(state["vocab"][group]) for group in groups})
    mappings = {kind: {identifier: index for index, identifier in enumerate(identifiers)}
                for kind, identifiers in nodes.items()}
    features = {}
    for kind in ("candidate", "job"):
        table = context["candidates" if kind == "candidate" else "jobs"]
        features[kind] = [profile_vector(table[identifier], kind, state) for identifier in nodes[kind]]
    for group in groups:
        features[group] = [hashed_vector([f"{group}:{token}"], state["feature_dim"]) for token in nodes[group]]
    edge_types, edges = [], {}
    rng = random.Random(seed)

    def add_relation(source, relation, destination, pairs):
        selected = sorted(set(tuple(pair) for pair in pairs))
        selected = [pair for pair in selected if rng.random() <= edge_retention]
        forward = (source, relation, destination)
        reverse = (destination, f"rev_{relation}", source)
        for edge_type, values in ((forward, selected), (reverse, [(right, left) for left, right in selected])):
            edge_types.append(list(edge_type))
            edges["|".join(edge_type)] = values

    pairs = context["past"] if message_pairs is None else message_pairs
    interaction_edges = []
    for row in pairs:
        if row["time"] >= context["cutoff"]:
            raise ValueError("Future interaction passed to graph builder")
        if row["candidate_id"] in mappings["candidate"] and row["job_id"] in mappings["job"]:
            interaction_edges.append((mappings["candidate"][row["candidate_id"]], mappings["job"][row["job_id"]]))
    add_relation("candidate", "interacted", "job", interaction_edges)
    for kind in ("candidate", "job"):
        table = context["candidates" if kind == "candidate" else "jobs"]
        relation_edges = defaultdict(list)
        for identifier in nodes[kind]:
            for group, members in attributes(table[identifier], kind, state["use_llm"], state["salary_bounds"]).items():
                if group in groups:
                    for member in members:
                        if member in mappings[group]:
                            relation_edges[group].append((mappings[kind][identifier], mappings[group][member]))
        for group in groups:
            relation = f"{'has' if kind == 'candidate' else 'requires'}_{group}"
            if group == "salary":
                relation = "has_current_salary" if kind == "candidate" else "offers_salary"
            if group == "location":
                relation = "located_in"
            add_relation(kind, relation, group, relation_edges[group])
    summary = {"graph_level": graph_level, "nodes": {kind: len(values) for kind, values in nodes.items()},
               "edges": {kind: len(values) for kind, values in edges.items()},
               "cutoff": context["cutoff"], "edge_retention": edge_retention,
               "schema_hash": digest(edge_types), "feature_dim": state["feature_dim"]}
    return {"nodes": nodes, "mappings": mappings, "features": features,
            "edge_types": edge_types, "edges": edges, "summary": summary}


def to_torch(graph, device="cpu"):
    import torch
    features = {kind: torch.tensor(values, dtype=torch.float32, device=device).reshape(-1, graph["summary"]["feature_dim"])
                for kind, values in graph["features"].items()}
    edges = {tuple(edge_type): torch.tensor(graph["edges"]["|".join(edge_type)], dtype=torch.long, device=device).reshape(-1, 2).t().contiguous()
             for edge_type in graph["edge_types"]}
    return features, edges
