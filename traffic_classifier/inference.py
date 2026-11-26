"""Load the chosen CNN once and classify packet prefixes through one model path."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from traffic_classifier.ml.dataset import encode_prefix
from traffic_classifier.ml.final_evaluation import DEFAULT_PROTOCOL
from traffic_classifier.ml.model_bundle import ModelBundle, load_model_bundle
from traffic_classifier.ml.packets import InvalidPackets
from traffic_classifier.ml.prefix_study import DEFAULT_OUTPUT
from traffic_classifier.vnat_groups import CLASSES


@dataclass(frozen=True)
class PacketInput:
    offsets_us: Sequence[int]
    sizes_bytes: Sequence[int]
    directions: Sequence[int]


@dataclass(frozen=True)
class Prediction:
    category: str
    scores: dict[str, float]
    model_id: str
    n: int
    observed_packets: int


@dataclass(frozen=True)
class Predictor:
    bundle: ModelBundle

    @property
    def model_id(self) -> str:
        return self.bundle.model_id

    @property
    def n(self) -> int:
        return self.bundle.normalizer.n

    def predict(self, packets: PacketInput) -> Prediction:
        return self.predict_many((packets,))[0]

    def predict_many(self, flows: Sequence[PacketInput]) -> list[Prediction]:
        if not flows:
            return []
        prefixes = [self._encode(packets) for packets in flows]
        features, masks = zip(*prefixes, strict=True)
        x = torch.from_numpy(np.stack(features))
        mask = torch.from_numpy(np.stack(masks))
        scores = self.bundle.logits(x, mask).softmax(dim=1)
        return [
            Prediction(
                CLASSES[winner],
                dict(zip(CLASSES, row, strict=True)),
                self.model_id,
                self.n,
                count,
            )
            for winner, row, count in zip(
                scores.argmax(dim=1).tolist(),
                scores.tolist(),
                mask.sum(dim=1).tolist(),
                strict=True,
            )
        ]

    def _encode(self, packets: PacketInput) -> tuple[np.ndarray, np.ndarray]:
        length = len(packets.offsets_us)
        if length != len(packets.sizes_bytes) or length != len(packets.directions):
            raise InvalidPackets("packet arrays must have the same length")
        if any(
            type(offset) is not int or not -(2**63) <= offset < 2**63
            for offset in packets.offsets_us[: self.n]
        ):
            raise InvalidPackets("packet offsets must be integer microseconds in Int64 range")
        return encode_prefix(packets.offsets_us, packets.sizes_bytes, packets.directions, self.n)


def load_predictor(
    protocol_path: Path = DEFAULT_PROTOCOL, study_dir: Path = DEFAULT_OUTPUT
) -> Predictor:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    n = protocol["selected_cnn_n"]
    bundle = load_model_bundle(study_dir / "cnn" / f"n{n}" / "model.pt", expected_n=n)
    if bundle.model_id != protocol["lengths"][str(n)]["cnn_model_id"]:
        raise ValueError("CNN bundle differs from the version recorded in the protocol.")
    return Predictor(bundle)
