import pytest

from src.core.tensor_trainer import _mean_reward


def test_empty_training_epoch_fails_instead_of_saving_zero_reward_tensor():
    with pytest.raises(RuntimeError, match="produced no rewards"):
        _mean_reward([], epoch=1)


def test_epoch_reward_uses_observed_questions():
    assert _mean_reward([0.25, 0.75], epoch=1) == 0.5
