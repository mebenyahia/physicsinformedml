"""Utility models for the mass-spring Physics Informed Neural Network demo.

This module keeps the neural network definitions separate from the notebooks so
that the same architectures can be reused in different experiments.  The code is
written with beginners in mind and contains detailed comments explaining every
step.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch import nn


def _get_activation(name: str) -> Callable[[], nn.Module]:
    """Return a constructor for the requested activation function.

    Parameters
    ----------
    name:
        Name of the activation function.  Supported values are ``"tanh"``,
        ``"relu"`` and ``"sigmoid"``.  The comparison is case insensitive.

    Returns
    -------
    Callable[[], nn.Module]
        A small function that, when called, creates the activation module.  The
        indirection allows us to build the final network sequentially without
        hardcoding a specific activation class.

    Raises
    ------
    ValueError
        If an unknown activation name is provided.
    """

    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "sigmoid":
        return nn.Sigmoid

    raise ValueError(
        "Unknown activation '{name}'. Please choose from 'tanh', 'relu', or 'sigmoid'."
        .format(name=name)
    )


class TimeMLP(nn.Module):
    """Simple multilayer perceptron that maps time ``t`` to displacement ``x``.

    The network receives a tensor of times with shape ``(batch_size, 1)`` and
    predicts the displacement of the mass at those times.  The architecture is
    intentionally small so that training is fast even on CPUs.

    Parameters
    ----------
    hidden_layers:
        Iterable with the number of neurons in each hidden layer.  A value like
        ``[64, 64]`` creates two hidden layers with 64 neurons each.
    activation:
        Name of the activation function to use between hidden layers.  See
        :func:`_get_activation` for available choices.
    """

    def __init__(self, hidden_layers: Sequence[int] = (64, 64), activation: str = "tanh") -> None:
        super().__init__()

        layers: List[nn.Module] = []
        act_ctor = _get_activation(activation)

        in_features = 1  # only time t is provided as input
        for hidden_units in hidden_layers:
            # Fully connected layer from the previous size to the new size
            layers.append(nn.Linear(in_features, hidden_units))
            # Non-linearity helps the network approximate curved functions
            layers.append(act_ctor())
            in_features = hidden_units

        # Final layer maps to a single displacement value
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the displacement for the provided times."""

        return self.model(t)


@dataclass
class ModelConfig:
    """Small helper dataclass collecting configuration options for the networks."""

    hidden_layers: Sequence[int] = (64, 64)
    activation: str = "tanh"


def build_baseline_model(config: Optional[ModelConfig] = None) -> TimeMLP:
    """Create a neural network for the baseline experiment.

    The baseline network only sees noisy measurements and tries to fit them.  We
    keep the architecture identical to the PINN to make comparisons fair.
    """

    config = config or ModelConfig()
    return TimeMLP(hidden_layers=config.hidden_layers, activation=config.activation)


def build_pinn_model(config: Optional[ModelConfig] = None) -> TimeMLP:
    """Create a neural network to be trained as a Physics Informed Neural Network.

    The architecture matches :func:`build_baseline_model`, but the training loop
    in the notebook will add an additional physics-based loss term.
    """

    config = config or ModelConfig()
    return TimeMLP(hidden_layers=config.hidden_layers, activation=config.activation)


__all__ = [
    "TimeMLP",
    "ModelConfig",
    "build_baseline_model",
    "build_pinn_model",
]
