"""Physics-Informed Machine Learning Package."""

from src.pinn_heat_equation import PINN, HeatEquationPINN, plot_solution

__version__ = "1.0.0"
__author__ = "mebenyahia"

__all__ = [
    "PINN",
    "HeatEquationPINN", 
    "plot_solution",
]
