import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cross_exp.exp_crossformer import Exp_crossformer


def make_experiment(root, batch_size):
    (root / "toy.csv").write_text(
        "date,value\n0,-1\n1,1\n2,2\n3,1\n4,1\n5,5\n", encoding="utf-8",
    )
    args = SimpleNamespace(
        root_path=str(root), data_path="toy.csv", data_split=[2, 1, 3],
        scale_statistic={"mean": np.array([0.0]), "std": np.array([1.0])},
        batch_size=batch_size, num_workers=0, use_gpu=False, use_multi_gpu=False,
        data_dim=1, in_len=2, out_len=1, seg_len=1, win_size=2,
        factor=1, d_model=4, d_ff=8, n_heads=1, e_layers=1, dropout=0.0,
        baseline=False,
    )
    with torch.random.fork_rng(devices=[]):
        experiment = Exp_crossformer(args)
    with torch.no_grad():
        for parameter in experiment.model.parameters():
            parameter.zero_()
    return experiment


@pytest.mark.parametrize("entry_point", ["test", "eval"])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_rmse_is_over_all_forecasts(tmp_path, monkeypatch, entry_point, batch_size):
    monkeypatch.chdir(tmp_path)
    experiment = make_experiment(tmp_path, batch_size)
    result = getattr(experiment, entry_point)("toy", save_pred=True)

    output = tmp_path / "results" / "toy"
    pred = np.load(output / "pred.npy")
    true = np.load(output / "true.npy")
    np.testing.assert_array_equal(pred, np.zeros((3, 1, 1)))
    np.testing.assert_array_equal(true, np.array([1, 1, 5]).reshape(3, 1, 1))

    scores = np.load(output / "metrics.npy")
    assert scores.shape == (5,)
    if entry_point == "eval":
        assert isinstance(result, tuple)
        np.testing.assert_array_equal(result, scores)
    else:
        assert result is None

    # Hand calculation over all three windows, independent of their batching.
    mean_square_error = (1 ** 2 + 1 ** 2 + 5 ** 2) / 3
    assert scores[[0, 1, 3, 4]] == pytest.approx([7 / 3, mean_square_error, 1, 1])
    assert scores[2] == pytest.approx(math.sqrt(mean_square_error))
