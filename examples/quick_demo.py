"""
Simple example demonstrating PINN training and visualization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pinn_heat_equation import HeatEquationPINN, plot_solution
import torch
import numpy as np

# Set random seed
torch.manual_seed(123)
np.random.seed(123)

print("=" * 60)
print("Quick PINN Demo - 1D Heat Equation")
print("=" * 60)

# Create solver
solver = HeatEquationPINN(alpha=0.1, layers=[2, 32, 32, 32, 1])

# Train with fewer epochs for quick demo
print("\nTraining PINN (quick demo with 2000 epochs)...")
solver.train(n_epochs=2000, n_pde=5000, verbose=True)

# Visualize
print("\nGenerating plots...")
plot_solution(solver, save_path='outputs/quick_demo.png')

print("\n" + "=" * 60)
print("Demo complete! Check outputs/quick_demo.png")
print("=" * 60)
