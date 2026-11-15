import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from jobrec.models import JobRecommender, RelationGate


@pytest.mark.parametrize("architecture", ["sage", "rgcn", "hgt"])
def test_forward_backward(architecture):
    metadata = (["candidate", "job"], [("candidate", "interacted", "job"), ("job", "rev_interacted", "candidate")])
    model = JobRecommender(architecture, metadata, input_dim=8, hidden=8, heads=2, dropout=0)
    features = {"candidate": torch.randn(3, 8), "job": torch.randn(4, 8)}
    edges = {metadata[1][0]: torch.tensor([[0, 1], [1, 2]]), metadata[1][1]: torch.tensor([[1, 2], [0, 1]])}
    embeddings = model(features, edges)
    scores = model.score_pairs(embeddings, torch.tensor([0, 1]), torch.tensor([1, 2]))
    loss = scores.square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())


@pytest.mark.parametrize("mode", ["uniform", "global", "profile", "history"])
def test_all_missing_gate_is_zero_and_finite(mode):
    gate = RelationGate(8, 8, mode)
    result = gate(torch.randn(2, 8), torch.zeros(2, 8), torch.zeros(2, 6, dtype=torch.bool))
    assert torch.isfinite(result).all()
    assert result.sum() == 0


def test_history_gate_receives_gradient():
    gate = RelationGate(8, 8, "history")
    history = torch.randn(2, 8, requires_grad=True)
    result = gate(torch.randn(2, 8), history, torch.ones(2, 6, dtype=torch.bool))
    (result * torch.arange(6)).sum().backward()
    assert history.grad is not None and history.grad.abs().sum() > 0


@pytest.mark.parametrize("architecture", ["sage", "rgcn", "hgt", "mf", "lightgcn"])
def test_train_checkpoint_roundtrip(tmp_path, architecture):
    from jobrec.data import make_temporal_split
    from jobrec.training import evaluate_checkpoint, train
    from jobrec.inference import Predictor
    candidates = [{"candidate_id": identifier, "create_date": "2020-01-01", "skills": "python", "zipcode": "01"} for identifier in ("a", "b")]
    jobs = [{"job_id": str(index), "create_date": "2020-01-01", "skills": "python;sql", "salary": str(index * 1000)} for index in range(1, 7)]
    pairs = [{"candidate_id": user, "job_id": job, "time": f"2021-01-{index:02}"}
             for index, (user, job) in enumerate((("a", "1"), ("b", "2"), ("a", "3"), ("b", "4"), ("a", "5"), ("b", "6")), 1)]
    bundle = {"candidates": candidates, "jobs": jobs, "pairs": pairs,
              "split": make_temporal_split(pairs, 0.5, 0.2), "manifest": {"dataset_hash": "toy"}}
    config = {"feature_dim": 8, "hidden": 8, "layers": 1, "heads": 2, "dropout": 0,
              "epochs": 1, "use_llm": False, "negatives": 1}
    gate_mode = "none" if architecture in {"mf", "lightgcn"} else "history"
    profile = train(bundle, config, architecture, tmp_path, gate_mode=gate_mode)
    assert profile["parameters"] > 0
    result, _, ranked = evaluate_checkpoint(bundle, tmp_path / "best.pt", "validation")
    assert result["users"] > 0
    prediction = Predictor(bundle, tmp_path / "best.pt", "validation").recommend("b")
    assert [row["job_id"] for row in prediction["recommendations"]] == ranked["b"]
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=True)
    assert not any(row in checkpoint["message_pairs"] for row in bundle["split"]["partitions"]["validation"])
