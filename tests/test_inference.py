import pytest
import torch

from traffic_classifier.inference import PacketInput, Predictor
from traffic_classifier.ml.cnn import PacketCNN
from traffic_classifier.ml.model_bundle import ModelBundle
from traffic_classifier.ml.normalization import Normalizer
from traffic_classifier.ml.packets import InvalidPackets
from traffic_classifier.vnat_groups import CLASSES


@pytest.fixture
def predictor():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = PacketCNN().eval()
    return Predictor(ModelBundle(model, Normalizer(8, (1.0, 2.0), (2.0, 3.0), "i", "s"), "model"))


def test_single_and_batch_predict_the_same_prefixes(predictor):
    short = PacketInput([0, 2000, 5250], [100, 300, 180], [0, 1, 0])
    long = PacketInput(list(range(9)), [100] * 8 + [999], [1] * 9)
    expected = predictor.bundle.packet_logits(
        short.offsets_us, short.sizes_bytes, short.directions
    ).softmax(dim=0)
    result, truncated = predictor.predict_many([short, long])
    assert tuple(result.scores) == CLASSES
    assert list(result.scores.values()) == pytest.approx(expected.tolist())
    assert result.category == predictor.predict(short).category
    assert result.observed_packets == 3
    assert (result.n, result.model_id) == (8, "model")
    assert truncated.observed_packets == 8
    assert truncated.scores == pytest.approx(
        predictor.predict(PacketInput(list(range(8)), [100] * 8, [1] * 8)).scores
    )


@pytest.mark.parametrize(
    ("packets", "message"),
    [
        (PacketInput([], [], []), "empty"),
        (PacketInput([0], [], [1]), "same length"),
        (PacketInput([0], [-1], [1]), "packet size"),
        (PacketInput([0], [120], [2]), "direction"),
        (PacketInput([1], [120], [1]), "relative packet time"),
        (PacketInput([0, 2**63], [10, 20], [1, 0]), "Int64 range"),
    ],
)
def test_invalid_prefix_is_rejected(predictor, packets, message):
    with pytest.raises(InvalidPackets, match=message):
        predictor.predict(packets)
