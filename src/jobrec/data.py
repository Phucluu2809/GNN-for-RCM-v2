import bisect
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


MISSING = {"", "nan", "null", "none"}
STAGE_FIELDS = (
    "spontaneous_application_date", "shortlist_date", "qualification_date",
    "resume_sent_to_company_date", "1st_interview_date", "2nd_interview_date",
    "3rd_interview_date", "4th_interview_date", "job_offer_proposed_date",
    "job_offer_accepted_date",
)


def missing(value):
    return str(value or "").strip().lower() in MISSING


def parse_date(value):
    if missing(value):
        return None
    return date.fromisoformat(value.strip()).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=None if path.suffix == ".gz" else 2, allow_nan=False)


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_raw(data_dir, jobs_file="jobs (1).csv"):
    tables, manifest = {}, {}
    required = {
        "candidates": ("candidates.csv", {"candidate_id", "create_date", "skills"}),
        "jobs": (jobs_file, {"job_id", "create_date", "skills"}),
        "history": ("history.csv", {"application_id", "candidate_id", "job_id", *STAGE_FIELDS}),
    }
    for name, (filename, columns) in required.items():
        path = Path(data_dir) / filename
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not columns.issubset(reader.fieldnames or []):
                raise ValueError(f"{filename}: missing columns {sorted(columns - set(reader.fieldnames or []))}")
            tables[name] = list(reader)
        manifest[name] = {"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return tables, manifest


def audit_data(tables):
    report = {}
    for name, rows in tables.items():
        if not rows:
            raise ValueError(f"Empty table: {name}")
        key = {"candidates": "candidate_id", "jobs": "job_id", "history": "application_id"}[name]
        identifiers = [row[key] for row in rows]
        if any(missing(identifier) for identifier in identifiers) or len(set(identifiers)) != len(rows):
            raise ValueError(f"Missing or duplicate primary key in {name}.{key}")
        invalid_dates = Counter()
        for row in rows:
            for column, value in row.items():
                if column.endswith("_date"):
                    try:
                        parse_date(value)
                    except ValueError:
                        invalid_dates[column] += 1
        report[name] = {
            "rows": len(rows), "columns": list(rows[0]),
            "missing": {column: sum(missing(row[column]) for row in rows) for column in rows[0]},
            "invalid_dates": dict(invalid_dates),
        }
    candidate_ids = {row["candidate_id"] for row in tables["candidates"]}
    job_ids = {row["job_id"] for row in tables["jobs"]}
    history = tables["history"]
    if any(row["candidate_id"] not in candidate_ids or row["job_id"] not in job_ids for row in history):
        raise ValueError("Orphan candidate/job foreign keys in history")
    pairs = {(row["candidate_id"], row["job_id"]) for row in history}
    user_jobs = defaultdict(set)
    for candidate_id, job_id in pairs:
        user_jobs[candidate_id].add(job_id)
    report["interactions"] = {
        "unique_pairs": len(pairs), "additional_pair_rows": len(history) - len(pairs),
        "users": len(user_jobs), "jobs": len({pair[1] for pair in pairs}),
        "jobs_per_user": dict(Counter(len(values) for values in user_jobs.values())),
        "last_stage_counts": dict(Counter(row.get("last_stage_reached", "") for row in history)),
        "sparsity": 1 - len(pairs) / (len(candidate_ids) * len(job_ids)),
    }
    return report


def build_events(history):
    events, first_pairs = [], {}
    for row in history:
        observed = []
        for stage in STAGE_FIELDS:
            timestamp = parse_date(row[stage])
            if timestamp:
                observed.append(timestamp)
                events.append({"candidate_id": row["candidate_id"], "job_id": row["job_id"],
                               "application_id": row["application_id"], "stage": stage, "time": timestamp})
        if not observed:
            raise ValueError(f"Application {row['application_id']} has no observed stage date")
        pair = (row["candidate_id"], row["job_id"])
        timestamp = min(observed)
        if pair not in first_pairs or timestamp < first_pairs[pair]["time"]:
            first_pairs[pair] = {"candidate_id": pair[0], "job_id": pair[1], "time": timestamp}
    key = lambda row: (row["time"], row["candidate_id"], row["job_id"])
    return sorted(events, key=key), sorted(first_pairs.values(), key=key)


def make_temporal_split(pairs, train_fraction=0.7, validation_fraction=0.15):
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("Fractions must leave nonempty train/validation/test periods")
    ordered = sorted(pairs, key=lambda row: row["time"])
    if len({row["time"] for row in ordered}) < 3:
        raise ValueError("Need at least three distinct event dates")
    cutoff1 = ordered[min(int(len(ordered) * train_fraction), len(ordered) - 1)]["time"]
    cutoff2 = ordered[min(int(len(ordered) * (train_fraction + validation_fraction)), len(ordered) - 1)]["time"]
    splits = {"train": [], "validation": [], "test": []}
    for row in ordered:
        partition = "train" if row["time"] < cutoff1 else "validation" if row["time"] < cutoff2 else "test"
        splits[partition].append(row)
    if not all(splits.values()):
        raise ValueError("Date ties produced an empty split; choose different fractions")
    return {"train_cutoff": cutoff1, "test_cutoff": cutoff2, "partitions": splits,
            "sha256": digest(splits)}


def available_profiles(rows, key, cutoff):
    available, excluded = {}, Counter()
    for row in rows:
        try:
            created = parse_date(row["create_date"])
        except ValueError:
            excluded["invalid_create_date"] += 1
            continue
        if created is None:
            excluded["missing_create_date"] += 1
        elif created >= cutoff:
            excluded["not_created_before_cutoff"] += 1
        else:
            available[row[key]] = row
    return available, dict(excluded)


def evaluation_context(bundle, partition):
    if partition not in {"validation", "test"}:
        raise ValueError("Evaluation partition must be validation or test")
    split = bundle["split"]
    cutoff = split["train_cutoff"] if partition == "validation" else split["test_cutoff"]
    candidates, excluded_candidates = available_profiles(bundle["candidates"], "candidate_id", cutoff)
    jobs, excluded_jobs = available_profiles(bundle["jobs"], "job_id", cutoff)
    past = [row for row in bundle["pairs"] if row["time"] < cutoff]
    seen = defaultdict(set)
    for row in past:
        seen[row["candidate_id"]].add(row["job_id"])
    positives, excluded = defaultdict(set), Counter()
    target = split["partitions"][partition]
    for row in target:
        if row["candidate_id"] not in candidates:
            excluded["candidate_unavailable"] += 1
        elif row["job_id"] not in jobs:
            excluded["job_unavailable"] += 1
        elif row["job_id"] in seen[row["candidate_id"]]:
            excluded["already_seen"] += 1
        else:
            positives[row["candidate_id"]].add(row["job_id"])
    return {"cutoff": cutoff, "candidates": candidates, "jobs": jobs, "past": past,
            "seen": seen, "positives": positives,
            "coverage": {"target_pairs": len(target), "eligible_pairs": sum(map(len, positives.values())),
                         "eligible_users": len(positives), "excluded_pairs": dict(excluded),
                         "excluded_candidates": excluded_candidates, "excluded_jobs": excluded_jobs}}


def prepare(config):
    tables, files = load_raw(config["data_dir"], config.get("jobs_file", "jobs (1).csv"))
    audit = audit_data(tables)
    invalid = {name: values["invalid_dates"] for name, values in audit.items() if "invalid_dates" in values and values["invalid_dates"]}
    if invalid:
        raise ValueError(f"Invalid dates must be corrected explicitly: {invalid}")
    events, pairs = build_events(tables["history"])
    split = make_temporal_split(pairs, config.get("train_fraction", 0.7), config.get("validation_fraction", 0.15))
    bundle = {"candidates": tables["candidates"], "jobs": tables["jobs"], "events": events,
              "pairs": pairs, "split": split,
              "manifest": {"version": "local_unverified", "files": files,
                           "dataset_hash": digest(files), "license": "CC BY-NC 4.0; verify local release",
                           "snapshot_warning": "Static profiles are not historical feature snapshots"}}
    output = Path(config["output_dir"])
    write_json(output / "bundle.json.gz", bundle)
    write_json(output / "manifest.json", bundle["manifest"])
    write_json(output / "audit.json", audit)
    write_json(output / "split.json", split)
    coverage = {partition: evaluation_context(bundle, partition)["coverage"] for partition in ("validation", "test")}
    write_json(output / "coverage.json", coverage)
    return bundle, audit, coverage


def eligible_negatives(candidate_id, timestamp, jobs, seen, rng, count):
    ordered = sorted((row["create_date"], identifier) for identifier, row in jobs.items())
    end = bisect.bisect_left([item[0] for item in ordered], timestamp)
    pool = [identifier for _, identifier in ordered[:end] if identifier not in seen.get(candidate_id, set())]
    if not pool:
        return []
    return rng.choices(pool, k=count)


class NegativeSampler:
    def __init__(self, context):
        ordered = sorted((row["create_date"], identifier) for identifier, row in context["jobs"].items())
        self.dates = [timestamp for timestamp, _ in ordered]
        self.identifiers = [identifier for _, identifier in ordered]
        self.seen = defaultdict(set)
        for row in context["past"]:
            self.seen[row["candidate_id"]].add(row["job_id"])

    def sample(self, candidate_id, timestamp, rng, count):
        if count < 1:
            raise ValueError("Negative count must be positive")
        end = bisect.bisect_left(self.dates, timestamp)
        if end == 0:
            return []
        samples = []
        for _ in range(count * 30):
            identifier = self.identifiers[rng.randrange(end)]
            if identifier not in self.seen[candidate_id]:
                samples.append(identifier)
                if len(samples) == count:
                    return samples
        pool = [identifier for identifier in self.identifiers[:end] if identifier not in self.seen[candidate_id]]
        return rng.choices(pool, k=count) if pool else []


def finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None
