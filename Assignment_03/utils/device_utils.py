import torch


def get_device(device_string):
    """
    Returns the device to be used for training.

    Args:
        device_string (str): Device string (cpu or cuda).

    Returns:
        torch.device: The device to be used for training.
    """
    if device_string == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_string == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    print("CUDA or MPS is not available. Using CPU.")
    return torch.device("cpu")
