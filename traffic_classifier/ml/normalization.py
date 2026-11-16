"""Fit and apply the packet feature scaler using only training packets."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from traffic_classifier.ml.dataset import DEFAULT_BUNDLE, Bundle, FlowDataset, load_bundle
from traffic_classifier.ml.vnat_split import DEFAULT_MANIFEST
from traffic_classifier.vnat_groups import PREFIX_LENGTHS

DEFAULT_DIR = Path("artifacts/normalization-v1")
SCALED_CHANNELS = (0, 2)


@dataclass(frozen=True)
class Normalizer:
    n: int
    mean: tuple[float, float]
    scale: tuple[float, float]
    import_id: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.n not in PREFIX_LENGTHS or len(self.mean) != 2 or len(self.scale) != 2:
            raise ValueError("Invalid normalizer shape or packet count.")
        if not all(math.isfinite(value) for value in (*self.mean, *self.scale)):
            raise ValueError("Normalizer must contain finite values.")
        if not all(value > 0 for value in self.scale):
            raise ValueError("Normalizer scales must be positive.")

    def transform(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (3, self.n) or mask.shape != (len(x), self.n):
            raise ValueError("Expected packet features B x 3 x N and mask B x N.")
        if mask.dtype != torch.bool or not x.is_floating_point():
            raise ValueError("Expected floating-point features and a Boolean mask.")
        output = x.masked_fill(~mask.unsqueeze(1), 0)
        for channel, mean, scale in zip(SCALED_CHANNELS, self.mean, self.scale, strict=True):
            output[:, channel] = (torch.log1p(output[:, channel]) - mean) / scale
        return output.masked_fill(~mask.unsqueeze(1), 0)

    def to_dict(self) -> dict:
        return {"version": 1, "method": "log1p_zscore", "fit_split": "train", **asdict(self)}

    @classmethod
    def from_dict(cls, config: dict) -> "Normalizer":
        if (config["version"], config["method"], config["fit_split"]) != (
            1,
            "log1p_zscore",
            "train",
        ):
            raise ValueError("Unsupported or non-training normalizer.")
        return cls(
            int(config["n"]),
            tuple(config["mean"]),
            tuple(config["scale"]),
            config["import_id"],
            config["manifest_sha256"],
        )


def fit_normalizer(bundle: Bundle, n: int) -> Normalizer:
    if bundle.split != "train":
        raise ValueError("Normalization can only be fitted on train.")
    dataset = FlowDataset(bundle, n)
    selected = dataset.features[:, SCALED_CHANNELS].transpose(1, 2)[dataset.mask].double()
    logged = torch.log1p(selected)
    scale = logged.std(dim=0, correction=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return Normalizer(
        n,
        tuple(logged.mean(dim=0).tolist()),
        tuple(scale.tolist()),
        bundle.import_id,
        bundle.manifest_sha256,
    )


def load_normalizer(path: Path, bundle: Bundle | None = None) -> Normalizer:
    normalizer = Normalizer.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if bundle is not None and (
        normalizer.import_id != bundle.import_id
        or normalizer.manifest_sha256 != bundle.manifest_sha256
    ):
        raise ValueError("Normalizer and packet bundle refer to different data.")
    return normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train = load_bundle(args.packages, "train", args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    for n in PREFIX_LENGTHS:
        normalizer = fit_normalizer(train, n)
        path = args.output / f"n{n}.json"
        path.write_text(json.dumps(normalizer.to_dict(), indent=2) + "\n", encoding="utf-8")
        dataset = FlowDataset(train, n)
        x, mask = dataset.features[:8], dataset.mask[:8]
        if not torch.equal(
            normalizer.transform(x, mask), load_normalizer(path, train).transform(x, mask)
        ):
            raise RuntimeError(f"Normalization round trip failed for N={n}.")
        print(f"N={n}: {dataset.mask.sum().item()} real train packets; {path}")


if __name__ == "__main__":
    main()
