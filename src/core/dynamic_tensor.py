"""Blueprint 5.1 - Query-Conditioned Dynamic Anisotropic Tensor W_A(Q).

Replaces the static heuristic alpha/beta/gamma and contiguous coordinate
slicing with a learned hypernetwork over query semantics and linked-entity
graph statistics. All graph statistics are min-max normalized to [0,1]
BEFORE mixing (fixes the scale-collapse defect by construction), and the
head predicts full-dimension weights (no contiguous block semantics).
"""
from __future__ import annotations

import torch
import torch.nn as nn

class QueryAdaptiveAnisotropicTensor(nn.Module):
    """Hypernetwork W_A(Q): full-dim learned weights; block stats in [0,1]."""

    def __init__(self, embed_dim: int = 768, feat_dim: int = 25, hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim + feat_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, embed_dim), nn.Softplus(),
        )
        self.component_head = nn.Linear(hidden, 3)  # a(Q) over {E, R, M}
        # adapter[:3] is (Linear -> GELU -> LayerNorm): the hidden state for
        # the component head. (adapter[:-1] also included the output Linear,
        # feeding a d-dim vector into the 3-way head - crashed at runtime.)
        self._backbone = nn.Sequential(*list(self.adapter)[:3])

    def forward(self, Q: torch.Tensor, graph_feats: torch.Tensor):
        """Q: (3, d) HCAD matrix; graph_feats: (feat_dim,) normalized stats.

        Returns (lambda, a): lambda in R^d with sum == 1; a is a 3-simplex.
        """
        z = torch.cat([Q.mean(dim=0), graph_feats], dim=-1)
        h = self._backbone(z)                    # through LayerNorm
        lam = self.adapter(z) + 1e-5
        lam = lam / lam.sum()                   # sum of lambda = 1, always
        a = torch.softmax(self.component_head(h), dim=-1)
        return lam, a

    @staticmethod
    def structural_similarity(Q: torch.Tensor, C: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Elementwise trace Tr(Q diag(lam) C^T) - no d x d matrix allocated."""
        return torch.sum(Q * lam.unsqueeze(0) * C)

def make_query_features(
    query_embedding: torch.Tensor,
    degree_norm: float,
    mean_conf: float,
    idf_sq: float,           # squashed to (0, 1)
    negation_flag: float,
    urgency_flag: float,
    type_histogram: list[float],
    relation_onehot: list[float],
) -> torch.Tensor:
    """Assemble the z(Q) feature vector; every component in [0, 1]."""
    stats = [max(0.0, min(1.0, x)) for x in
             [degree_norm, mean_conf, idf_sq, negation_flag, urgency_flag]]
    feats = stats + list(type_histogram) + list(relation_onehot)
    return torch.tensor(feats, dtype=torch.float32)


def save_tensor(model: "QueryAdaptiveAnisotropicTensor", path=None) -> str:
    """Persist the trained hypernetwork (only trainable weights)."""
    import torch
    from src.config import TENSOR_MODEL_PATH
    path = path or TENSOR_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return str(path)


def load_tensor(path=None) -> "QueryAdaptiveAnisotropicTensor":
    """Load a trained hypernetwork; raises if no checkpoint exists."""
    import torch
    from src.config import TENSOR_MODEL_PATH
    path = path or TENSOR_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No trained tensor at {path}")
    model = QueryAdaptiveAnisotropicTensor()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
