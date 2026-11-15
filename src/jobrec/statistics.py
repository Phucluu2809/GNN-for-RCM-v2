import random


def paired_bootstrap(left, right, metric="ndcg@20", repetitions=2000, seed=42):
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparison requires the same nonempty user cohort")
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    differences = [right[identifier][metric] - left[identifier][metric] for identifier in sorted(left)]
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(repetitions))
    return {"metric": metric, "users": len(differences), "difference_right_minus_left": sum(differences) / len(differences),
            "ci95": [estimates[int(0.025 * repetitions)], estimates[min(repetitions - 1, int(0.975 * repetitions))]],
            "repetitions": repetitions, "seed": seed,
            "scope": "user sampling uncertainty for these runs, not variation across training seeds"}
