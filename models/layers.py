import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # During evaluation, do nothing
        if not self.training or self.p == 0:
            return x

        # Create dropout mask
        mask = (torch.rand_like(x) > self.p).float()

        # Inverted dropout scaling
        return x * mask / (1 - self.p)

