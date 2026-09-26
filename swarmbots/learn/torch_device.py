import torch


def as_device(device: torch.device | str) -> torch.device:
    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    if resolved_device.type == "cuda" and resolved_device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if resolved_device.type == "cpu":
        return torch.device("cpu")
    return resolved_device
