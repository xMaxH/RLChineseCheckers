"""NN inference helpers.

For the verify and early stages we run inference in-process: a single net
on the GPU is shared by sequential self-play games. The interface
(`make_nn_eval`) returns a callable matching the MCTS contract:

    nn_eval(boards: np.ndarray, globs: np.ndarray) -> (policy_logits, values)

Both as numpy arrays.

A multi-process inference server with cross-worker batching can be added
later behind the same interface; nothing else needs to change.
"""

from typing import Callable, Tuple
import numpy as np
import torch

from .net import AZNet


def make_nn_eval(net: AZNet, device: torch.device) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return a closure that evaluates a batch through `net` on `device`."""
    net.eval()

    def nn_eval(boards: np.ndarray, globs: np.ndarray):
        with torch.no_grad():
            b = torch.from_numpy(boards).to(device, non_blocking=True)
            g = torch.from_numpy(globs).to(device, non_blocking=True)
            pol, val = net(b, g)
        return pol.cpu().numpy(), val.cpu().numpy()

    return nn_eval


def load_model(path: str, device: torch.device) -> AZNet:
    net = AZNet().to(device)
    sd = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    return net
