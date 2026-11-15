import math

from .features import experience_interval


def constraint_policy(preferences, verified_job):
    checks = {}
    locations = preferences.get("required_locations")
    if locations:
        location = verified_job.get("location")
        checks["location"] = "unknown" if location is None else "pass" if location in locations else "fail"
    if preferences.get("experience_years") is not None and preferences.get("enforce_experience"):
        actual = experience_interval(str(preferences["experience_years"]))
        required = experience_interval(verified_job.get("minimum_experience"))
        checks["experience"] = "unknown" if actual is None or required is None else "fail" if actual[1] < required[0] else "pass" if actual[0] >= required[0] else "unknown"
    minimum_salary = preferences.get("minimum_salary")
    if minimum_salary is not None:
        maximum = verified_job.get("salary_max")
        same_unit = (verified_job.get("salary_currency") == preferences.get("salary_currency")
                     and verified_job.get("salary_period") == preferences.get("salary_period")
                     and preferences.get("salary_currency") is not None
                     and preferences.get("salary_period") is not None)
        checks["salary"] = "unknown" if maximum is None or not same_unit else "fail" if maximum < minimum_salary else "pass"
    overall = "fail" if "fail" in checks.values() else "unknown" if "unknown" in checks.values() or not checks else "pass"
    return {"status": overall, "checks": checks}


def rerank(scored_jobs, preferences, verified_jobs, top_k=20):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    result = []
    seen = set()
    for identifier, score in sorted(scored_jobs, key=lambda pair: (-pair[1], pair[0])):
        if identifier in seen:
            raise ValueError("Duplicate job in scored list")
        seen.add(identifier)
        if not math.isfinite(score):
            raise ValueError("Nonfinite recommendation score")
        policy = constraint_policy(preferences, verified_jobs.get(identifier, {}))
        if policy["status"] != "fail":
            result.append({"job_id": identifier, "score": score, "policy": policy})
        if len(result) == top_k:
            break
    return result
