"""A compact one-dimensional CNN for masked packet prefixes."""

import argparse

import torch
from torch import nn

ARCHITECTURE = "packet-cnn-v1"


class PacketCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 3))

    def conv_features(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 3 or mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError("Expected features B x 3 x N and mask B x N.")
        if mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise ValueError("Every sequence needs a Boolean mask with a real packet.")
        padding = ~mask.unsqueeze(1)
        first = torch.relu(self.conv1(x.masked_fill(padding, 0))).masked_fill(padding, 0)
        second = torch.relu(self.conv2(first)).masked_fill(padding, 0)
        return first, second

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, encoded = self.conv_features(x, mask)
        pooled = encoded.sum(dim=2) / mask.sum(dim=1, keepdim=True)
        return self.head(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.manual_seed(42)
    model = PacketCNN().eval()
    x = torch.rand(2, 3, 8)
    mask = torch.arange(8) < torch.tensor([[3], [8]])
    with torch.inference_mode():
        first, second = model.conv_features(x, mask)
        logits = model(x, mask)
        padded = x.masked_fill(~mask.unsqueeze(1), 1000)
        torch.testing.assert_close(model(padded, mask), logits)
        torch.testing.assert_close(model(x[:1, :, :3], mask[:1, :3]), logits[:1])
    params = sum(p.numel() for p in model.parameters())
    assert params == 2307
    print(f"x={tuple(x.shape)} conv1={tuple(first.shape)} conv2={tuple(second.shape)}")
    print(f"logits={tuple(logits.shape)} trainable parameters={params}")


if __name__ == "__main__":
    main()
