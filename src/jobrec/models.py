import torch
from torch import nn
from torch.nn import functional as functional
from torch_geometric.nn import HeteroConv, HGTConv, RGCNConv, SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self, architecture, metadata, input_dim=128, hidden=64, layers=2, heads=4, dropout=0.2):
        super().__init__()
        if architecture not in {"sage", "rgcn", "hgt"}:
            raise ValueError("architecture must be sage, rgcn or hgt")
        if layers < 1 or hidden < 1 or input_dim < 1 or heads < 1:
            raise ValueError("Model dimensions/layers/heads must be positive")
        if architecture == "hgt" and hidden % heads:
            raise ValueError("HGT hidden must be divisible by heads")
        self.architecture = architecture
        self.node_types, self.edge_types = metadata
        self.dropout = dropout
        self.projections = nn.ModuleDict({kind: nn.Linear(input_dim, hidden) for kind in self.node_types})
        self.layers = nn.ModuleList()
        for _ in range(layers):
            if architecture == "sage":
                layer = HeteroConv({edge_type: SAGEConv((hidden, hidden), hidden, root_weight=False, bias=False)
                                    for edge_type in self.edge_types}, aggr="mean")
            elif architecture == "rgcn":
                layer = RGCNConv(hidden, hidden, len(self.edge_types), num_bases=min(4, len(self.edge_types)))
            else:
                layer = HGTConv(hidden, hidden, metadata, heads=heads)
            self.layers.append(layer)

    def forward(self, features, edges):
        embeddings = {kind: functional.relu(self.projections[kind](features[kind])) for kind in self.node_types}
        offsets, lengths, total = {}, {}, 0
        if self.architecture == "rgcn":
            for kind in self.node_types:
                offsets[kind], lengths[kind] = total, embeddings[kind].shape[0]
                total += lengths[kind]
            homogeneous_edges, relations = [], []
            for relation_id, edge_type in enumerate(self.edge_types):
                shift = edges[edge_type].new_tensor([[offsets[edge_type[0]]], [offsets[edge_type[2]]]])
                homogeneous_edges.append(edges[edge_type] + shift)
                relations.append(torch.full((edges[edge_type].shape[1],), relation_id, dtype=torch.long, device=edges[edge_type].device))
            edge_index = torch.cat(homogeneous_edges, dim=1)
            edge_types = torch.cat(relations)
        for layer in self.layers:
            if self.architecture == "rgcn":
                homogeneous = torch.cat([embeddings[kind] for kind in self.node_types])
                messages = layer(homogeneous, edge_index, edge_types)
                messages = {kind: messages[offsets[kind]:offsets[kind] + lengths[kind]] for kind in self.node_types}
            else:
                messages = layer(embeddings, edges)
            next_embeddings = {}
            for kind, previous in embeddings.items():
                message = messages.get(kind)
                combined = previous if message is None else previous + message
                next_embeddings[kind] = functional.dropout(functional.relu(combined), self.dropout, self.training)
            embeddings = next_embeddings
        return {kind: functional.normalize(values, dim=-1) for kind, values in embeddings.items()}


class RelationGate(nn.Module):
    def __init__(self, feature_dim, hidden=64, mode="history"):
        super().__init__()
        if mode not in {"uniform", "global", "profile", "history"}:
            raise ValueError("Unknown relation gate mode")
        self.mode = mode
        if mode == "global":
            self.logits = nn.Parameter(torch.zeros(6))
        elif mode in {"profile", "history"}:
            self.network = nn.Sequential(nn.Linear(feature_dim * 2 + 6, hidden), nn.ReLU(), nn.Linear(hidden, 6))

    def forward(self, profiles, history, mask):
        if self.mode == "uniform":
            logits = torch.zeros_like(mask, dtype=profiles.dtype)
        elif self.mode == "global":
            logits = self.logits.expand(profiles.shape[0], -1)
        else:
            history_input = history if self.mode == "history" else torch.zeros_like(history)
            logits = self.network(torch.cat((profiles, history_input, mask.to(profiles.dtype)), dim=-1))
        valid = mask.any(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(~mask, -1e9)
        return torch.softmax(safe_logits, dim=-1) * mask.to(logits.dtype) * valid.to(logits.dtype)


class CollaborativeEncoder(nn.Module):
    def __init__(self, architecture, node_ids, hidden, layers):
        super().__init__()
        self.architecture, self.layers = architecture, layers
        self.lookup = {kind: {identifier: index + 1 for index, identifier in enumerate(identifiers)}
                       for kind, identifiers in node_ids.items()}
        self.tables = nn.ModuleDict({kind: nn.Embedding(len(identifiers) + 1, hidden, padding_idx=0)
                                     for kind, identifiers in node_ids.items()})
        for table in self.tables.values():
            nn.init.normal_(table.weight, std=0.1)
            with torch.no_grad():
                table.weight[0].zero_()
        self.current_nodes = node_ids

    def forward(self, features, edges):
        initial = {}
        for kind in ("candidate", "job"):
            indices = torch.tensor([self.lookup[kind].get(identifier, 0) for identifier in self.current_nodes[kind]],
                                   dtype=torch.long, device=features[kind].device)
            initial[kind] = self.tables[kind](indices)
        if self.architecture == "mf":
            return initial
        candidate_count = initial["candidate"].shape[0]
        current = torch.cat((initial["candidate"], initial["job"]))
        edge_index = edges[("candidate", "interacted", "job")]
        source = torch.cat((edge_index[0], edge_index[1] + candidate_count))
        destination = torch.cat((edge_index[1] + candidate_count, edge_index[0]))
        degree = torch.bincount(source, minlength=current.shape[0]).to(current.dtype).clamp_min(1)
        normalization = (degree[source] * degree[destination]).rsqrt()
        average = current
        for _ in range(self.layers):
            propagated = torch.zeros_like(current)
            propagated.index_add_(0, destination, current[source] * normalization.unsqueeze(-1))
            current = propagated
            average = average + current
        average = average / (self.layers + 1)
        return {"candidate": average[:candidate_count], "job": average[candidate_count:]}


class JobRecommender(nn.Module):
    def __init__(self, architecture, metadata, input_dim=128, hidden=64, layers=2, heads=4, dropout=0.2, gate_mode="none", node_ids=None):
        super().__init__()
        if architecture in {"mf", "lightgcn"}:
            if gate_mode != "none" or node_ids is None:
                raise ValueError("Collaborative baselines need node_ids and gate_mode=none")
            self.encoder = CollaborativeEncoder(architecture, node_ids, hidden, layers)
        else:
            self.encoder = GNNEncoder(architecture, metadata, input_dim, hidden, layers, heads, dropout)
        self.gate_mode = gate_mode
        self.gate = None if gate_mode == "none" else RelationGate(input_dim, hidden, gate_mode)

    def forward(self, features, edges):
        return self.encoder(features, edges)

    def set_node_ids(self, nodes):
        if isinstance(self.encoder, CollaborativeEncoder):
            self.encoder.current_nodes = nodes

    def score_pairs(self, embeddings, candidates, jobs, profiles=None, history=None, masks=None, compatibilities=None):
        scores = (embeddings["candidate"][candidates] * embeddings["job"][jobs]).sum(dim=-1)
        if self.gate is not None:
            alpha = self.gate(profiles[candidates], history[candidates], masks[candidates])
            scores = scores + (alpha * compatibilities).sum(dim=-1)
        return scores
