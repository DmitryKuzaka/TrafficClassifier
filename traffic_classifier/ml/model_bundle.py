"""Export and load a self-contained packet CNN for CPU inference."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from traffic_classifier.import_vnat import RELEASE_SHA256
from traffic_classifier.ml.cnn import ARCHITECTURE, PacketCNN
from traffic_classifier.ml.dataset import encode_prefix
from traffic_classifier.ml.normalization import DEFAULT_DIR as NORMALIZATION_DIR
from traffic_classifier.ml.normalization import Normalizer, load_normalizer
from traffic_classifier.ml.vnat_split import DEFAULT_MANIFEST
from traffic_classifier.vnat_groups import CLASSES, PREFIX_LENGTHS

FORMAT_VERSION = 1
INPUT_CHANNELS = ("size_bytes", "relative_direction", "gap_ms")
DEFAULT_OUTPUT = Path("artifacts/cnn-bundles-v1")


@dataclass(frozen=True)
class ModelBundle:
    model: PacketCNN
    normalizer: Normalizer
    model_id: str

    @torch.inference_mode()
    def logits(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalizer.transform(features, mask), mask)

    def packet_logits(
        self, offsets_us: Sequence[int], sizes: Sequence[int], directions: Sequence[int]
    ) -> torch.Tensor:
        if not (len(offsets_us) == len(sizes) == len(directions)):
            raise ValueError("Packet arrays must have the same length.")
        features, mask = encode_prefix(offsets_us, sizes, directions, self.normalizer.n)
        return self.logits(torch.from_numpy(features)[None], torch.from_numpy(mask)[None])[0]


def model_identifier(metadata: dict, state: dict[str, torch.Tensor]) -> str:
    digest = sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return f"{ARCHITECTURE}-n{metadata['n']}-{digest.hexdigest()}"


def unpack_model(payload: dict, *, expected_n: int | None = None) -> ModelBundle:
    metadata = payload["metadata"]
    if metadata["format_version"] != FORMAT_VERSION or metadata["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported CNN bundle format or architecture.")
    if tuple(metadata["classes"]) != CLASSES or tuple(metadata["input_channels"]) != INPUT_CHANNELS:
        raise ValueError("Bundle class map or channel order does not match this version.")
    if metadata["dataset_sha256"] != RELEASE_SHA256:
        raise ValueError("Unexpected source dataset version.")

    n = metadata["n"]
    if (
        type(n) is not int
        or n not in PREFIX_LENGTHS
        or (expected_n is not None and n != expected_n)
    ):
        raise ValueError("CNN bundle has an incompatible packet count.")
    normalizer = Normalizer.from_dict(metadata["normalization"])
    if (
        normalizer.n != n
        or normalizer.import_id != metadata["import_id"]
        or normalizer.manifest_sha256 != metadata["manifest_sha256"]
    ):
        raise ValueError("CNN bundle normalization does not match its metadata.")

    # nn.Module initializers use the RNG even though weights are immediately replaced.
    with torch.random.fork_rng(devices=[]):
        model = PacketCNN()
    state = payload["state_dict"]
    if any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float32
        for value in state.values()
    ):
        raise ValueError("Expected float32 CNN weights.")
    model.load_state_dict(state, strict=True)
    if payload["model_id"] != model_identifier(metadata, state):
        raise ValueError("CNN bundle contents do not match its model ID.")
    model.eval()
    return ModelBundle(model, normalizer, payload["model_id"])


def load_model_bundle(path: Path, *, expected_n: int | None = None) -> ModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return unpack_model(payload, expected_n=expected_n)


def export_model_bundle(
    checkpoint_path: Path,
    normalizer_path: Path,
    output: Path,
    *,
    expected_n: int | None = None,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    normalizer = load_normalizer(normalizer_path)
    metadata = {
        "format_version": FORMAT_VERSION,
        "architecture": checkpoint["architecture"],
        "n": checkpoint["config"]["n"],
        "classes": list(CLASSES),
        "input_channels": list(INPUT_CHANNELS),
        "dataset_release": "VNAT_Dataframe_release_1.h5",
        "dataset_sha256": RELEASE_SHA256,
        "split_manifest": DEFAULT_MANIFEST.name,
        "import_id": checkpoint["import_id"],
        "manifest_sha256": checkpoint["manifest_sha256"],
        "best_epoch": checkpoint["best_epoch"],
        "normalization": normalizer.to_dict(),
    }
    state = checkpoint["state_dict"]
    model_id = model_identifier(metadata, state)
    payload = {"metadata": metadata, "state_dict": state, "model_id": model_id}
    unpack_model(payload, expected_n=expected_n)

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return model_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=PREFIX_LENGTHS, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint or Path(f"artifacts/cnn-experiments-v1/n{args.n}/checkpoint.pt")
    normalizer = args.normalizer or NORMALIZATION_DIR / f"n{args.n}.json"
    output = args.output or DEFAULT_OUTPUT / f"n{args.n}.pt"
    model_id = export_model_bundle(checkpoint, normalizer, output, expected_n=args.n)
    print(f"{model_id} -> {output}")


if __name__ == "__main__":
    main()
