import hashlib
import heapq
import math
import random
from collections import Counter, defaultdict

from .data import NegativeSampler, digest, evaluation_context
from .features import attributes, compatibility
from .graph import training_context
from .metrics import aggregate_metrics


def content_tokens(row, kind, state):
    groups = attributes(row, kind, state["use_llm"], state["salary_bounds"])
    return {f"{group}:{value}" for group in ("skill", "industry") for value in groups[group]}


def pair_vector(candidate, job, state):
    values, masks = compatibility(candidate, job, state)
    return values + [float(value) for value in masks]


def fit_logistic(bundle, state, seed):
    from sklearn.linear_model import LogisticRegression
    context = training_context(bundle)
    sampler, rng = NegativeSampler(context), random.Random(seed)
    features, labels = [], []
    for row in context["past"]:
        candidate = context["candidates"].get(row["candidate_id"])
        positive = context["jobs"].get(row["job_id"])
        if candidate is None or positive is None:
            continue
        negatives = sampler.sample(row["candidate_id"], row["time"], rng, 5)
        if not negatives:
            continue
        features.append(pair_vector(candidate, positive, state))
        labels.append(1)
        for identifier in negatives:
            features.append(pair_vector(candidate, context["jobs"][identifier], state))
            labels.append(0)
    if len(set(labels)) < 2:
        raise ValueError("Logistic baseline needs both positive and sampled negative pairs")
    model = LogisticRegression(max_iter=500, random_state=seed)
    model.fit(features, labels)
    return model


def evaluate_baseline(bundle, state, model="popular", partition="validation", seed=42, max_users=None):
    if model not in {"random", "popular", "content", "logreg"}:
        raise ValueError("Baseline must be random, popular, content or logreg")
    classifier = fit_logistic(bundle, state, seed) if model == "logreg" else None
    context = evaluation_context(bundle, partition)
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users must be positive")
    users = sorted(context["positives"])
    if max_users is not None:
        users = random.Random(seed).sample(users, min(len(users), max_users))
    popularity = Counter(row["job_id"] for row in bundle["split"]["partitions"]["train"])
    catalog = sorted(context["jobs"])
    popular_order = sorted(catalog, key=lambda identifier: (-popularity[identifier], identifier))
    inverted, job_lengths = defaultdict(list), {}
    if model == "content":
        for identifier, row in context["jobs"].items():
            values = content_tokens(row, "job", state)
            job_lengths[identifier] = math.sqrt(len(values)) or 1.0
            for value in values:
                inverted[value].append(identifier)
    recommendations = {}
    for candidate_id in users:
        seen = context["seen"].get(candidate_id, set())
        if model == "random":
            user_seed = int(hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest(), 16)
            pool = [identifier for identifier in catalog if identifier not in seen]
            ranked = random.Random(user_seed).sample(pool, min(20, len(pool)))
        elif model == "popular":
            ranked = [identifier for identifier in popular_order if identifier not in seen][:20]
        elif model == "logreg":
            pool = [identifier for identifier in catalog if identifier not in seen]
            values = [pair_vector(context["candidates"][candidate_id], context["jobs"][identifier], state) for identifier in pool]
            scores = classifier.decision_function(values) if pool else []
            ranked = [identifier for identifier, _ in sorted(zip(pool, scores), key=lambda pair: (-pair[1], pair[0]))[:20]]
        else:
            values = content_tokens(context["candidates"][candidate_id], "candidate", state)
            overlap = Counter(identifier for value in sorted(values) for identifier in inverted[value] if identifier not in seen)
            normalizer = math.sqrt(len(values)) or 1.0
            ranked = heapq.nsmallest(20, overlap, key=lambda identifier: (-overlap[identifier] / (normalizer * job_lengths[identifier]), identifier))
            selected = set(ranked)
            ranked.extend(identifier for identifier in popular_order if identifier not in seen and identifier not in selected)
            ranked = ranked[:20]
        recommendations[candidate_id] = ranked
    positives = {identifier: context["positives"][identifier] for identifier in users}
    warm_users = {row["candidate_id"] for row in context["past"]}
    result, per_user = aggregate_metrics(recommendations, positives, len(catalog), warm_users)
    result.update({"model": model, "partition": partition, "seed": seed, "coverage": context["coverage"],
                   "evaluation": "full_catalog", "max_users": max_users,
                   "preprocessing_hash": digest(state), "feature_schema_version": state["feature_schema_version"],
                   "use_llm": state["use_llm"],
                   "dataset_hash": bundle["manifest"]["dataset_hash"], "split_hash": bundle["split"]["sha256"]})
    if classifier is not None:
        result["classifier"] = {"coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
                                "features": [f"compat_{group}" for group in ("skill", "industry", "experience", "location", "salary", "education")]
                                            + [f"observed_{group}" for group in ("skill", "industry", "experience", "location", "salary", "education")]}
    return result, per_user, recommendations
