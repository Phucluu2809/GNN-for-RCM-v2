import math
from collections import defaultdict


def user_metrics(ranked, positives, top_k):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not positives:
        raise ValueError("Ranking metrics require at least one eligible positive")
    if len(ranked) != len(set(ranked)):
        raise ValueError("Duplicate recommendations")
    hits = [index + 1 for index, identifier in enumerate(ranked[:top_k]) if identifier in positives]
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(top_k, len(positives)) + 1))
    return {f"recall@{top_k}": len(hits) / len(positives), f"precision@{top_k}": len(hits) / top_k,
            f"ndcg@{top_k}": dcg / ideal, f"mrr@{top_k}": 1 / hits[0] if hits else 0.0,
            f"fill@{top_k}": min(len(ranked), top_k) / top_k}


def aggregate_metrics(recommendations, positives, catalog_size, warm_users=None, ks=(5, 10, 20)):
    per_user, groups = {}, defaultdict(list)
    for candidate_id, relevant in positives.items():
        if not relevant:
            continue
        ranked = recommendations.get(candidate_id, [])
        metrics = {}
        for cutoff in ks:
            metrics.update(user_metrics(ranked, relevant, cutoff))
        per_user[candidate_id] = metrics
        group = "warm" if candidate_id in (warm_users or set()) else "cold"
        groups[group].append(metrics)
    def average(rows):
        return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}
    result = {"users": len(per_user), "metrics": average(list(per_user.values())),
              "cohorts": {group: {"users": len(rows), "metrics": average(rows)} for group, rows in groups.items()}}
    for cutoff in ks:
        unique = {identifier for ranked in recommendations.values() for identifier in ranked[:cutoff]}
        result["metrics"][f"coverage@{cutoff}"] = len(unique) / catalog_size if catalog_size else 0.0
    return result, per_user
