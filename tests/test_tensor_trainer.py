import pytest

from src.core.tensor_trainer import _mean_reward


def test_empty_training_epoch_fails_instead_of_saving_zero_reward_tensor():
    with pytest.raises(RuntimeError, match="produced no rewards"):
        _mean_reward([], epoch=1)


def test_epoch_reward_uses_observed_questions():
    assert _mean_reward([0.25, 0.75], epoch=1) == 0.5


def test_listwise_loss_produces_real_gradients():
    """The replacement loss must produce genuine gradients (the collapsed
    REINFORCE backpropagated ~1e-7 floating-point noise)."""
    import torch

    from src.core.dynamic_tensor import ChannelWeightAdapter

    torch.manual_seed(0)
    adapter = ChannelWeightAdapter(feat_dim=25)
    feats = torch.rand(25)
    S = torch.tensor([[0.9, 0.1, 0.2],
                      [0.2, 0.9, 0.1],
                      [0.3, 0.3, 0.9]])
    y = torch.tensor([1.0, 0.0, 0.0])
    a = adapter(feats)
    phi = S @ a
    logp = torch.log_softmax(phi / 0.15, dim=-1)
    loss = -logp[y > 0].mean()
    loss.backward()
    norm = sum(p.grad.norm().item() for p in adapter.parameters()
               if p.grad is not None)
    assert norm > 1e-6, f"gradient norm {norm} is numerically dead"


def test_old_simplex_score_function_is_identically_zero():
    """Characterization of the historical zero-sum collapse: the old
    trainer's score function summed simplex-normalized differences, which
    is exactly zero - and its gradient w.r.t. the parameters is zero too.
    Guards against re-introducing simplex-sum score functions."""
    import torch

    torch.manual_seed(0)
    logits = torch.randn(768, requires_grad=True)
    mu_lam = torch.softmax(logits, dim=-1)          # simplex: sum == 1
    eps = torch.randn(768)
    lam_s = mu_lam + 0.10 * eps
    lam_s = lam_s / lam_s.sum()                     # projected back: sum == 1
    gl = ((lam_s - mu_lam) / (0.10 ** 2)).sum()
    # float32 summation over 768 terms leaves ~1e-5 rounding residue;
    # a live score would be O(1/sigma^2) - the point is it is dead, not
    # bit-exact.
    assert abs(float(gl.detach())) < 1e-3, "simplex-sum score must be ~zero"
    gl.backward()
    assert logits.grad is None or float(logits.grad.norm()) < 1e-4, \
        "the old score function carries no gradient"


def test_adapter_learns_discriminative_channel():
    """Given a pool where gold candidates separate only on the relation
    channel, training must upweight that channel above the others."""
    import torch

    from src.core.dynamic_tensor import ChannelWeightAdapter

    torch.manual_seed(0)
    adapter = ChannelWeightAdapter(feat_dim=25)
    opt = torch.optim.Adam(adapter.parameters(), lr=0.05)
    feats = torch.rand(25)
    # 6 candidates: gold (rows 0,1) score high on channel 1 (relation),
    # distractors score high on channels 0 and 2.
    S = torch.tensor([
        [0.2, 0.9, 0.1],
        [0.3, 0.8, 0.2],
        [0.9, 0.2, 0.1],
        [0.8, 0.3, 0.2],
        [0.1, 0.2, 0.9],
        [0.2, 0.3, 0.8],
    ])
    gold = torch.tensor([True, True, False, False, False, False])

    first_loss = last_loss = None
    for step in range(300):
        a = adapter(feats)
        phi = S @ a
        logp = torch.log_softmax(phi / 0.15, dim=-1)
        loss = -logp[gold].mean()
        if step == 0:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()

    a_final = adapter(feats)
    assert last_loss < first_loss, "loss must decrease"
    assert a_final[1] > a_final[0] and a_final[1] > a_final[2], \
        f"adapter must favor the discriminative channel, got {a_final}"
    assert float(a_final[1]) > 0.4, f"relation weight must dominate: {a_final}"


def test_train_tensor_end_to_end_with_fake_retriever(tmp_path, monkeypatch):
    """Full loop: warm-up capture -> listwise step -> checkpoint saved and
    reloadable as a callable (lam, a) with isotropic lambda."""
    import numpy as np
    import torch

    import src.core.dynamic_tensor as dt
    from src.core.tensor_trainer import train_tensor

    ckpt = tmp_path / "dynamic_tensor.pt"
    monkeypatch.setattr("src.config.TENSOR_MODEL_PATH", ckpt)
    # Unit tests must not depend on the process-wide NER singleton: real
    # scispacy loads take ~25 s and can transiently fail (degrading the
    # singleton and tripping the trainer's feature-parity guard, which is
    # correct in production but is test pollution here).
    from types import SimpleNamespace
    monkeypatch.setattr("src.core.ner_extractor.get_biomedical_ner",
                        lambda: SimpleNamespace(degraded=False))

    class FakeRetriever:
        def __init__(self):
            self._tensor = None
            self._tensor_override = None
            self._last_tensor_inputs = None
            self._last_channel_sims = None
            self._last_candidate_pool = None
            self.calls = 0

        def retrieve(self, question):
            self.calls += 1
            assert self._tensor_override is not None, \
                "warm-up must run under the isotropic override"
            lam, a = self._tensor_override
            assert abs(float(lam.sum()) - 1.0) < 1e-5
            self._last_tensor_inputs = (
                torch.zeros(3, 768), torch.rand(25))
            # gold candidates separate on the relation channel; the sims
            # rows and the PRE-GATE candidate pool must stay index-aligned
            # exactly as the real retriever guarantees them.
            self._last_channel_sims = np.array(
                [[0.2, 0.9, 0.1], [0.9, 0.2, 0.1], [0.1, 0.2, 0.9]],
                dtype=np.float32)
            # _ranked_relevance reads candidates as dicts (document key),
            # mirroring the real retriever's output contract.
            self._last_candidate_pool = [
                {"document": "gold text about the association"},
                {"document": "distractor one"},
                {"document": "distractor two"}]
            return list(self._last_candidate_pool)

    fake = FakeRetriever()
    dev = [{"question": "What is the association?",
            "reference": "gold text about the association",
            "pool": "dev"} for _ in range(3)]
    result = train_tensor(fake, dev, epochs=2, lr=0.05)

    assert ckpt.exists(), "checkpoint must be written"
    assert result["epochs"], "telemetry must be recorded"
    assert result["epochs"][-1]["mean_ndcg_reward"] > 0.5, \
        "gold must outrank distractors after training"
    assert result["epochs"][-1]["mean_grad_norm"] > 1e-6, \
        "gradients must be alive, not collapsed noise"

    loaded = dt.load_channel_adapter(ckpt)
    lam, a = loaded(torch.zeros(3, 768), torch.rand(25))
    assert abs(float(lam.sum()) - 1.0) < 1e-5, "lambda stays isotropic"
    assert abs(float(a.sum()) - 1.0) < 1e-5, "channel weights stay on the simplex"
    assert float(a[1]) > 0.4, "learned weights must favor the gold channel"


def test_train_tensor_refuses_non_dev_pools():
    from src.core.tensor_trainer import train_tensor

    test_q = [{"question": "q", "reference": "r", "pool": "test"}]
    with pytest.raises(ValueError, match="non-dev"):
        train_tensor(retriever=object(), dev_questions=test_q, epochs=1)


def test_misaligned_sims_and_pool_skip_instead_of_crashing(tmp_path, monkeypatch):
    """Regression for the first Kaggle training crash (IndexError: sims
    had pre-gate rows, relevance had post-gate contexts). Misalignment
    must skip the question loudly; an epoch with nothing usable must
    fail via the empty-rewards guard, never crash."""
    import numpy as np
    import torch

    from src.core.tensor_trainer import train_tensor

    monkeypatch.setattr("src.config.TENSOR_MODEL_PATH",
                        tmp_path / "dynamic_tensor.pt")
    from types import SimpleNamespace
    monkeypatch.setattr("src.core.ner_extractor.get_biomedical_ner",
                        lambda: SimpleNamespace(degraded=False))

    class MisalignedRetriever:
        def __init__(self):
            self._tensor = None
            self._tensor_override = None
            self._last_tensor_inputs = None
            self._last_channel_sims = None
            self._last_candidate_pool = None

        def retrieve(self, question):
            self._tensor_override = (
                torch.full((768,), 1.0 / 768), torch.tensor([0.3, 0.5, 0.2]))
            self._last_tensor_inputs = (
                torch.zeros(3, 768), torch.rand(25))
            # sims claim 28 candidates (pre-gate), pool has 5 (post-gate)
            self._last_channel_sims = np.random.rand(28, 3).astype(np.float32)
            self._last_candidate_pool = [
                {"document": f"candidate {i}"} for i in range(5)]
            return list(self._last_candidate_pool)

    dev = [{"question": "q", "reference": "candidate 1", "pool": "dev"}]
    with pytest.raises(RuntimeError, match="produced no rewards"):
        train_tensor(MisalignedRetriever(), dev, epochs=1)
