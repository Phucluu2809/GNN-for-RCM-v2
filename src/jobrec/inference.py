import torch

from .data import evaluation_context
from .features import FEATURE_SCHEMA_VERSION, GROUPS, compatibility
from .graph import build_graph, to_torch
from .models import JobRecommender
from .ranking import rerank
from .training import gate_inputs


class Predictor:
    def __init__(self, bundle, checkpoint_path, partition="test", device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["preprocessors"].get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Feature schema changed; retrain this checkpoint")
        if checkpoint["dataset_hash"] != bundle["manifest"]["dataset_hash"] or checkpoint["split_hash"] != bundle["split"]["sha256"]:
            raise ValueError("Checkpoint does not match prepared dataset/split")
        self.device = device
        self.state = checkpoint["preprocessors"]
        self.context = evaluation_context(bundle, partition)
        self.graph = build_graph(self.context, self.state, checkpoint["config"].get("graph_level", "G2"))
        self.features, edges = to_torch(self.graph, device)
        self.model = JobRecommender(metadata=checkpoint["metadata"], **checkpoint["model_arguments"]).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.set_node_ids(self.graph["nodes"])
        with torch.no_grad():
            self.embeddings = self.model(self.features, edges)
        self.history, self.masks = gate_inputs(self.graph, self.context, self.state, device)

    @torch.no_grad()
    def recommend(self, candidate_id, top_k=20, session_job_ids=(), weights=None, preferences=None, verified_jobs=None):
        if candidate_id not in self.graph["mappings"]["candidate"]:
            raise ValueError("Candidate is not available at the selected cutoff")
        known_jobs = self.graph["mappings"]["job"]
        if any(identifier not in known_jobs for identifier in session_job_ids):
            raise ValueError("Session event references an unavailable job")
        index = self.graph["mappings"]["candidate"][candidate_id]
        past_ids = {row["job_id"] for row in self.context["past"] if row["candidate_id"] == candidate_id and row["job_id"] in known_jobs}
        history_ids = sorted(past_ids | set(session_job_ids))
        history = self.history[index:index + 1]
        if session_job_ids:
            history = self.features["job"][[known_jobs[identifier] for identifier in history_ids]].mean(dim=0, keepdim=True)
        mask = self.masks[index:index + 1]
        alpha = None
        if weights is not None:
            if len(weights) != 6 or any(not isinstance(value, (int, float)) or value < 0 or not torch.isfinite(torch.tensor(float(value))) for value in weights):
                raise ValueError("Provide six finite, nonnegative relation weights")
            alpha = torch.tensor([weights], device=self.device, dtype=torch.float32) * mask
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        elif self.model.gate is not None:
            alpha = self.model.gate(self.features["candidate"][index:index + 1], history, mask)
        jobs = [identifier for identifier in self.graph["nodes"]["job"] if identifier not in past_ids and identifier not in session_job_ids]
        scored = []
        for start in range(0, len(jobs), 1024):
            batch = jobs[start:start + 1024]
            scores = (self.embeddings["candidate"][index] * self.embeddings["job"][[known_jobs[identifier] for identifier in batch]]).sum(dim=-1)
            if alpha is not None:
                values = torch.tensor([compatibility(self.context["candidates"][candidate_id], self.context["jobs"][identifier], self.state)[0] for identifier in batch], device=self.device)
                scores += (values * alpha).sum(dim=-1)
            scored.extend(zip(batch, scores.cpu().tolist()))
        recommendations = rerank(scored, preferences or {}, verified_jobs or {}, top_k)
        weights_out = dict(zip(GROUPS, alpha[0].cpu().tolist())) if alpha is not None else None
        return {"recommendations": recommendations, "weights": weights_out,
                "mode": "manual_preference_scenario" if weights is not None else self.model.gate_mode,
                "catalog_size": len(jobs), "history_jobs": len(history_ids),
                "warning": "JTH fields are not verified hard constraints; unknown is not fail"}
