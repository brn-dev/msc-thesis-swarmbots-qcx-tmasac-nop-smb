import torch

class StickyActionDist:
    def _init_stickiness_buffer(self) -> None:
        self.register_buffer("_stickiness", torch.tensor(0.0, dtype=torch.float32))

    def _set_stickiness_buffer(self, stickiness: float) -> None:
        self._stickiness.fill_(float(stickiness))

    def _stickiness_tensor(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self._stickiness.to(device=device, dtype=dtype)

    def set_stickiness(self, stickiness: float) -> None:
        if not (0.0 <= stickiness < 1.0):
            raise ValueError(f"stickiness must be in [0, 1), got {stickiness}.")
        self._set_stickiness_buffer(stickiness)

    def get_stickiness(self) -> float:
        return float(self._stickiness.item())
