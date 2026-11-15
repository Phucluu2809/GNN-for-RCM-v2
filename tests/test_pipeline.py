import math
import random

import pytest

from jobrec.data import available_profiles, build_events, evaluation_context, make_temporal_split, STAGE_FIELDS
from jobrec.features import attributes, compatibility, experience_interval, fit_preprocessors, hashed_vector, normalize_location
from jobrec.graph import build_graph, disjoint_edges, training_context
from jobrec.metrics import user_metrics
from jobrec.ranking import constraint_policy, rerank


@pytest.fixture
def bundle():
    candidates = [{"candidate_id": identifier, "create_date": "2020-01-01", "skills": "python", "zipcode": "01"}
                  for identifier in ("a", "b", "c")]
    jobs = [{"job_id": str(index), "create_date": "2020-01-01", "skills": "python;sql", "salary": str(index * 10000)}
            for index in range(1, 7)]
    pairs = [{"candidate_id": candidate, "job_id": job, "time": timestamp}
             for candidate, job, timestamp in [("a", "1", "2021-01-01"), ("b", "2", "2021-01-02"),
                                                ("c", "3", "2021-01-03"), ("a", "2", "2021-01-04"),
                                                ("b", "4", "2021-01-05"), ("c", "5", "2021-01-06")]]
    split = make_temporal_split(pairs, 0.5, 0.2)
    return {"candidates": candidates, "jobs": jobs, "pairs": pairs, "events": [], "split": split,
            "manifest": {"dataset_hash": "toy"}}


def test_first_pair_dates_ignore_outcome_and_merge_applications():
    history = []
    for application, timestamp in (("one", "2020-02-01"), ("two", "2020-03-01")):
        row = {field: "" for field in STAGE_FIELDS}
        row.update(application_id=application, candidate_id="a", job_id="1",
                   shortlist_date=timestamp, end_of_process_date="2019-01-01")
        history.append(row)
    events, pairs = build_events(history)
    assert len(events) == 2
    assert pairs == [{"candidate_id": "a", "job_id": "1", "time": "2020-02-01"}]


def test_split_ties_and_no_overlap():
    rows = [{"candidate_id": str(index), "job_id": "j", "time": f"2020-01-{index // 2 + 1:02}"} for index in range(12)]
    split = make_temporal_split(rows, 0.5, 0.25)
    dates = [{row["time"] for row in rows} for rows in split["partitions"].values()]
    assert not dates[0] & dates[1] and not dates[1] & dates[2]
    with pytest.raises(ValueError):
        make_temporal_split(rows, 0.9, 0.2)


def test_availability_is_strict_and_does_not_infer_missing_date():
    rows = [{"id": "a", "create_date": ""}, {"id": "b", "create_date": "2020-01-02"},
            {"id": "c", "create_date": "2020-01-01"}]
    profiles, excluded = available_profiles(rows, "id", "2020-01-02")
    assert list(profiles) == ["c"]
    assert excluded["missing_create_date"] == 1


def test_metrics_hand_calculated_and_short_list():
    result = user_metrics(["irrelevant", "positive"], {"positive", "other"}, 20)
    assert result["recall@20"] == 0.5
    assert result["precision@20"] == 0.05
    assert result["mrr@20"] == 0.5
    assert result["fill@20"] == 0.1
    assert result["ndcg@20"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    with pytest.raises(ValueError):
        user_metrics(["j", "j"], {"j"}, 20)
    with pytest.raises(ValueError):
        user_metrics([], set(), 20)


def test_preprocessing_fit_does_not_see_future_salary(bundle):
    bundle["jobs"].append({"job_id": "future", "create_date": "2030-01-01", "skills": "futureonly", "salary": "99999999"})
    state = fit_preprocessors(bundle, {"use_llm": False})
    assert "futureonly" not in state["vocab"]["skill"]
    assert max(state["salary_bounds"]) <= 60000
    assert all(math.isfinite(value) for value in hashed_vector(["python"], 16))
    assert attributes({"skills": "_rare_skill_;Python;python"}, "candidate")["skill"] == {"python"}


def test_disjoint_supervision_and_reverse_edges(bundle):
    context = training_context(bundle)
    messages, supervision = disjoint_edges(context, 0.5, 42)
    keys = lambda rows: {(row["candidate_id"], row["job_id"]) for row in rows}
    assert not keys(messages) & keys(supervision)
    graph = build_graph(context, fit_preprocessors(bundle, {}), message_pairs=messages)
    forward = graph["edges"]["candidate|interacted|job"]
    reverse = graph["edges"]["job|rev_interacted|candidate"]
    assert {(right, left) for left, right in forward} == set(reverse)
    for row in supervision:
        assert (graph["mappings"]["candidate"][row["candidate_id"]], graph["mappings"]["job"][row["job_id"]]) not in forward


def test_graph_rejects_future_interactions(bundle):
    context = training_context(bundle)
    with pytest.raises(ValueError, match="Future"):
        build_graph(context, fit_preprocessors(bundle, {}), message_pairs=bundle["split"]["partitions"]["test"])


def test_pruning_reverse_edges_remain_symmetric(bundle):
    graph = build_graph(training_context(bundle), fit_preprocessors(bundle, {}), edge_retention=0.5)
    for source, relation, destination in graph["edge_types"]:
        if not relation.startswith("rev_"):
            forward = graph["edges"][f"{source}|{relation}|{destination}"]
            reverse = graph["edges"][f"{destination}|rev_{relation}|{source}"]
            assert {(right, left) for left, right in forward} == set(reverse)


def test_context_excludes_future_job_and_keeps_denominator_report(bundle):
    bundle["jobs"][4]["create_date"] = "2030-01-01"
    context = evaluation_context(bundle, "test")
    assert "5" not in context["jobs"]
    assert context["coverage"]["excluded_pairs"]["job_unavailable"] == 1
    assert context["coverage"]["target_pairs"] == 2


def test_policy_requires_explicit_verified_data():
    preference = {"minimum_salary": 50000, "salary_currency": "EUR", "salary_period": "annual"}
    assert constraint_policy(preference, {"salary": 20000})["status"] == "unknown"
    assert constraint_policy(preference, {"salary_max": 20000, "salary_currency": "EUR", "salary_period": "annual"})["status"] == "fail"
    assert constraint_policy({}, {})["status"] == "unknown"
    assert experience_interval("+6 years") == (6, math.inf)
    ranked = rerank([("one", 1), ("two", 0.5)], {"required_locations": ["HCM"]}, {"one": {"location": "HN"}}, top_k=20)
    assert [row["job_id"] for row in ranked] == ["two"]


def test_negative_sampling_obeys_creation_and_seen(bundle):
    from jobrec.data import eligible_negatives
    jobs = {row["job_id"]: row for row in bundle["jobs"]}
    jobs["6"]["create_date"] = "2030-01-01"
    values = eligible_negatives("a", "2021-01-01", jobs, {"a": {"1", "2"}}, random.Random(42), 200)
    assert values and not set(values) & {"1", "2", "6"}


def test_paired_bootstrap_requires_same_users():
    from jobrec.statistics import paired_bootstrap
    left = {"a": {"ndcg@20": 0.1}, "b": {"ndcg@20": 0.1}}
    right = {"a": {"ndcg@20": 0.3}, "b": {"ndcg@20": 0.3}}
    result = paired_bootstrap(left, right, repetitions=100)
    assert result["difference_right_minus_left"] == pytest.approx(0.2)
    assert result["ci95"] == pytest.approx([0.2, 0.2])
    with pytest.raises(ValueError):
        paired_bootstrap(left, {"other": {"ndcg@20": 0.3}})


def test_location_numeric_serialization_does_not_break_relations():
    assert normalize_location("75.0") == normalize_location("75")
    assert normalize_location("1.0") == "01"
    assert normalize_location("2a") == "2A"
    assert normalize_location("") is None
    state = {"use_llm": False, "salary_bounds": []}
    values, masks = compatibility({"zipcode": "75.0"}, {"zipcode": "75"}, state)
    assert values[3] == 1.0 and masks[3]
