import json

import numpy as np
import pytest
import torch

from traffic_classifier.ml.cnn import ARCHITECTURE, PacketCNN
from traffic_classifier.ml.dataset import encode_prefix
from traffic_classifier.ml.model_bundle import export_model_bundle, load_model_bundle
from traffic_classifier.ml.normalization import Normalizer


def example_model() -> PacketCNN:
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = PacketCNN()
        for parameter in model.parameters():
            parameter.zero_()
        # The masking test supplies sizes only. The bundle test exercises all input channels.
        model.conv1.weight[0, :, 1] = torch.tensor([1.0, 2.0, 3.0])
        model.conv1.weight[0, 0, 2] = 1
        model.conv1.bias[0] = 1
        model.conv2.weight[0, 0, 1:] = 1
        model.conv2.bias[0] = 1
        model.head[0].weight[0, 0] = 1
        model.head[3].weight[:, 0] = torch.tensor([1.0, 2.0, -1.0])
    return model.eval()


def test_encode_prefix_preserves_channel_order_and_real_zero_values():
    features, mask = encode_prefix([0, 2000, 5250], [100, 300, 180], [0, 1, 0], n=8)

    expected = np.array(
        [
            [100, 300, 180, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 2, 3.25, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(features, expected, strict=True)
    np.testing.assert_array_equal(
        mask, np.array([True, True, True, False, False, False, False, False]), strict=True
    )


def test_cnn_logits_depend_only_on_real_packets():
    model = example_model()
    features = torch.zeros(2, 3, 8)
    features[:, 0, 0] = 2
    features[1, 0, 1] = 4
    mask = torch.arange(8) < torch.tensor([[1], [2]])
    dirty_padding = features.masked_fill(~mask.unsqueeze(1), 1000)

    with torch.inference_mode():
        logits = model(features, mask)
        assert logits.shape == (2, 3)
        torch.testing.assert_close(logits, torch.tensor([[4.0, 8.0, -4.0], [9.5, 19.0, -9.5]]))
        torch.testing.assert_close(model(dirty_padding, mask), logits)
        torch.testing.assert_close(model(features[:1, :, :1], mask[:1, :1]), logits[:1])


def test_bundle_round_trip_keeps_logits_and_weights(tmp_path, monkeypatch):
    model = example_model()
    normalizer = Normalizer(
        n=8, mean=(1.0, 2.0), scale=(2.0, 3.0), import_id="test", manifest_sha256="split"
    )
    expected_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    checkpoint = tmp_path / "checkpoint.pt"
    normalization = tmp_path / "n8.json"
    output = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "config": {"n": 8},
            "import_id": normalizer.import_id,
            "manifest_sha256": normalizer.manifest_sha256,
            "best_epoch": 1,
            "state_dict": expected_weights,
        },
        checkpoint,
    )
    normalization.write_text(json.dumps(normalizer.to_dict()), encoding="utf-8")

    model_id = export_model_bundle(checkpoint, normalization, output, expected_n=8)
    checkpoint.unlink()
    normalization.unlink()
    monkeypatch.chdir(tmp_path)
    restored = load_model_bundle(output, expected_n=8)
    assert restored.model_id == model_id
    assert not restored.model.training

    packets = ([0, 2000, 5250], [100, 300, 180], [0, 1, 0])
    features, mask = encode_prefix(*packets, n=8)
    x = torch.from_numpy(features).unsqueeze(0).requires_grad_()
    mask = torch.from_numpy(mask).unsqueeze(0)
    with torch.inference_mode():
        expected_logits = model(normalizer.transform(x, mask), mask)
    with torch.enable_grad():
        actual_logits = restored.logits(x, mask)
        packet_logits = restored.packet_logits(*packets)
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(packet_logits, expected_logits[0])
    assert not actual_logits.requires_grad
    assert not packet_logits.requires_grad
    assert all(parameter.grad is None for parameter in restored.model.parameters())
    torch.testing.assert_close(restored.model.state_dict(), expected_weights, rtol=0, atol=0)

    with pytest.raises(ValueError, match="incompatible packet count"):
        load_model_bundle(output, expected_n=16)
