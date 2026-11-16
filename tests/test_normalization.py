import json

import numpy as np
import pytest
import torch

from traffic_classifier.ml.dataset import MAX_PACKETS, Bundle
from traffic_classifier.ml.normalization import Normalizer, fit_normalizer


def example_bundle(split: str) -> Bundle:
    features = np.zeros((1, 3, MAX_PACKETS), dtype=np.float32)
    features[0, :, :2] = [[0, 8], [1, 0], [0, 3]]
    mask = np.zeros((1, MAX_PACKETS), dtype=bool)
    mask[0, :2] = True
    return Bundle(features, mask, np.array([0]), np.array(["one"]), split, "manifest", "import")


def test_fit_ignores_padding_and_round_trips():
    bundle = example_bundle("train")
    normalizer = fit_normalizer(bundle, 8)
    x = torch.from_numpy(bundle.features[:, :, :8])
    mask = torch.from_numpy(bundle.mask[:, :8])
    normalized = normalizer.transform(x, mask)
    restored = Normalizer.from_dict(json.loads(json.dumps(normalizer.to_dict())))
    assert torch.equal(normalized, restored.transform(x, mask))
    assert torch.allclose(normalized[0, 0, :2], torch.tensor([-1.0, 1.0]))
    assert torch.allclose(normalized[0, 2, :2], torch.tensor([-1.0, 1.0]))
    assert torch.equal(normalized[0, 1], x[0, 1])
    assert torch.count_nonzero(normalized[:, :, 2:]) == 0


def test_fit_rejects_validation():
    with pytest.raises(ValueError, match="only be fitted on train"):
        fit_normalizer(example_bundle("validation"), 8)
