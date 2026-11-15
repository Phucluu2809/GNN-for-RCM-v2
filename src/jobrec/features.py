import hashlib
import math
import re
from collections import defaultdict

from .data import available_profiles, finite_float, missing


GROUPS = ("skill", "industry", "experience", "location", "salary", "education")
FEATURE_SCHEMA_VERSION = 1


def tokens(value):
    return {part.strip().lower() for part in str(value or "").split(";")
            if not missing(part) and part.strip().lower() != "_rare_skill_"}


def normalize_location(value):
    if missing(value):
        return None
    normalized = str(value).strip().upper()
    numeric = re.fullmatch(r"(\d+)(?:\.0+)?", normalized)
    return str(int(numeric.group(1))).zfill(2) if numeric else normalized


def attributes(row, kind, use_llm=True, salary_bounds=()):
    groups = {group: set() for group in GROUPS}
    groups["skill"] = tokens(row.get("skills"))
    groups["industry"] = {f"area:{value}" for value in tokens(row.get("expertise_area"))}
    groups["industry"].update(f"category:{value}" for value in tokens(row.get("job_category")))
    groups["experience"] = tokens(row.get("years_experience"))
    location = normalize_location(row.get("zipcode"))
    groups["location"] = {location} if location else set()
    salary = finite_float(row.get("actual_salary" if kind == "candidate" else "salary"))
    if salary is not None:
        groups["salary"] = {f"annual_eur:{sum(salary > boundary for boundary in salary_bounds)}"}
    if use_llm:
        for column in ("llm_hard_skills", "llm_programming_languages", "llm_tools_technologies"):
            groups["skill"].update(tokens(row.get(column)))
        groups["education"] = tokens(row.get("llm_highest_diploma" if kind == "candidate" else "llm_required_lowest_diploma"))
    return groups


def fit_preprocessors(bundle, config):
    cutoff = bundle["split"]["train_cutoff"]
    jobs, _ = available_profiles(bundle["jobs"], "job_id", cutoff)
    candidates, _ = available_profiles(bundle["candidates"], "candidate_id", cutoff)
    salaries = sorted(number for row in jobs.values() if (number := finite_float(row.get("salary"))) is not None)
    bounds = sorted({salaries[min(int(len(salaries) * quantile), len(salaries) - 1)]
                     for quantile in (0.25, 0.5, 0.75)}) if salaries else []
    state = {"salary_bounds": bounds, "feature_dim": config.get("feature_dim", 128),
             "feature_schema_version": FEATURE_SCHEMA_VERSION,
             "use_llm": config.get("use_llm", True), "vocab": {}, "fit_cutoff": cutoff}
    vocab = defaultdict(set)
    for kind, profiles in (("candidate", candidates), ("job", jobs)):
        for row in profiles.values():
            for group, values in attributes(row, kind, state["use_llm"], bounds).items():
                vocab[group].update(values)
    state["vocab"] = {group: sorted(vocab[group]) for group in GROUPS}
    return state


def hashed_vector(values, dimension):
    vector = [0.0] * dimension
    for value in sorted(set(values)):
        hashed = hashlib.blake2b(value.encode(), digest_size=8).digest()
        index = int.from_bytes(hashed[:4], "little") % dimension
        vector[index] += 1.0 if hashed[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def profile_vector(row, kind, state):
    groups = attributes(row, kind, state["use_llm"], state["salary_bounds"])
    values = [f"{group}:{value}" for group, members in groups.items() for value in members]
    values.extend(f"missing:{group}" for group, members in groups.items() if not members)
    values.extend(f"contract:{value}" for value in tokens(row.get("contract_type")))
    return hashed_vector(values, state["feature_dim"])


def compatibility(candidate, job, state):
    left = attributes(candidate, "candidate", state["use_llm"], state["salary_bounds"])
    right = attributes(job, "job", state["use_llm"], state["salary_bounds"])
    values, masks = [], []
    for group in GROUPS:
        observed = bool(left[group] and right[group])
        masks.append(observed)
        values.append(len(left[group] & right[group]) / len(right[group]) if observed else 0.0)
    return values, masks


def experience_interval(value):
    if missing(value):
        return None
    numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if not numbers:
        return None
    if len(numbers) >= 2:
        return min(numbers), max(numbers)
    return (numbers[0], math.inf) if "+" in str(value) else (numbers[0], numbers[0])
