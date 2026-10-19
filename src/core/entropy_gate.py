"""Blueprint 5.3 - Entropy-regulated bidirectional soft gating.

Replaces the discontinuous Heaviside with a differentiable sigmoid gate whose
cutoff couples to the generator's live token entropy (rho in [0,1] from a
short probe pass). High uncertainty loosens the gate (more support); high
parametric confidence tightens it (less distraction). The soft temperature
removes the sigma -> 0 discontinuity, and abstention is explicit.
"""
from __future__ import annotations

import math

import numpy as np

class EntropyRegulatedDualGate:
    def __init__(
        self,
        base_k: float = 1.0,
        floor: float = 0.10,
        rho0: float = 0.5,
        kappa: float = 0.6,
        temp_min: float = 1e-3,
        max_items: int = 8,
    ) -> None:
        self.k, self.floor, self.rho0, self.kappa = base_k, floor, rho0, kappa
        self.tmin, self.kmax = temp_min, max_items

    def gate(self, phi: np.ndarray, rho: float) -> np.ndarray:
        """phi: candidate scores; rho: normalized generator uncertainty in [0,1].

        Returns a boolean keep-mask. Empty output = abstention (parametric mode).
        """
        phi = np.asarray(phi, dtype=np.float64)
        if phi.size == 0:
            return np.zeros_like(phi, dtype=bool)
        if not np.isfinite(phi).all():
            raise ValueError("Non-finite candidate scores supplied to the gate")
        survivors = phi[phi >= self.floor]
        if survivors.size == 0:
            return np.zeros_like(phi, dtype=bool)   # abstain -> parametric mode
        mu, sd = survivors.mean(), max(survivors.std(), self.tmin)
        tau = max(self.floor, mu + (self.k - self.kappa * (rho - self.rho0)) * sd)
        m = 1.0 / (1.0 + np.exp(-(phi - tau) / sd))  # differentiable Heaviside
        keep = np.zeros_like(phi, dtype=bool)
        for i in np.argsort(-m):
            if m[i] < 0.5 or keep.sum() >= self.kmax:
                break
            keep[i] = True
        return keep

class UncertaintyMonitor:
    """Streams token entropies; fires re-retrieval on spikes."""

    def __init__(self, vocab_log_scale: float = 11.0, spike_z: float = 3.0, window: int = 24):
        self.H: list[float] = []
        self.spike_z, self.logV, self.window = spike_z, vocab_log_scale, window

    def update(self, logprobs: np.ndarray) -> bool:
        """Feed one step's token logprobs; returns True on an entropy spike."""
        p = np.exp(logprobs - logprobs.max())
        p = p / p.sum()
        self.H.append(float(-(p * np.log(p + 1e-12)).sum()))
        if len(self.H) >= self.window:
            h = np.asarray(self.H[-self.window:])
            mu, sd = h.mean(), max(h.std(), 1e-3)
            return bool(self.H[-1] > mu + self.spike_z * sd)
        return False

    @property
    def rho(self) -> float:
        """Normalized mean entropy in [0, 1]; 0.5 prior before any evidence."""
        if not self.H:
            return 0.5
        return float(np.clip(np.mean(self.H) / self.logV, 0.0, 1.0))
