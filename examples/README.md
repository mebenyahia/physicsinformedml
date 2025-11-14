# Examples Directory

This directory contains various examples demonstrating the use of Physics-Informed Neural Networks.

## Available Examples

### 1. Quick Demo (`quick_demo.py`)

A fast demonstration that trains a PINN in about 20-30 seconds.

**Run:**
```bash
python examples/quick_demo.py
```

**What it does:**
- Trains a PINN with 2000 epochs
- Generates visualization plots
- Saves output to `outputs/quick_demo.png`

**Good for:** Quick verification that everything works

---

### 2. Interactive Jupyter Notebook (`pinn_demo.ipynb`)

A comprehensive, interactive tutorial with detailed explanations.

**Run:**
```bash
jupyter notebook examples/pinn_demo.ipynb
```

**What it includes:**
- Step-by-step explanation of PINNs
- Mathematical background
- Code walkthrough
- Multiple visualizations
- Error analysis
- Time evolution plots

**Good for:** Learning and experimentation

---

## Creating Your Own Examples

### Basic Template

```python
import sys
sys.path.append('../src')

from pinn_heat_equation import HeatEquationPINN, plot_solution
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create and train solver
solver = HeatEquationPINN(alpha=0.1)
solver.train(n_epochs=5000)

# Visualize
plot_solution(solver, save_path='my_result.png')
```

### Custom Training

```python
# Adjust hyperparameters
solver = HeatEquationPINN(
    alpha=0.15,                   # Thermal diffusivity
    layers=[2, 100, 100, 100, 1]  # Network architecture
)

# Train with custom parameters
solver.train(
    n_epochs=10000,    # More epochs
    n_pde=15000,       # More collocation points
    lr=5e-4,           # Lower learning rate
    verbose=True
)
```

### Custom Visualization

```python
import matplotlib.pyplot as plt

# Plot at specific time
x = np.linspace(0, 1, 200)[:, None]
t = np.ones_like(x) * 0.5  # t = 0.5

u_pred = solver.predict(x, t)
u_exact = solver.exact_solution(x, t)

plt.figure(figsize=(10, 6))
plt.plot(x, u_pred, label='PINN', linewidth=2)
plt.plot(x, u_exact, '--', label='Exact', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x, 0.5)')
plt.title('Solution at t=0.5')
plt.legend()
plt.grid(True)
plt.savefig('custom_plot.png')
plt.show()
```

## Tips for Running Examples

### Memory Issues
If you run out of memory, reduce:
```python
solver.train(
    n_pde=5000,   # Fewer collocation points
    n_bc=100,     # Fewer boundary points
    n_ic=100      # Fewer initial points
)
```

### Speed Up Training
For faster training:
```python
# Smaller network
solver = HeatEquationPINN(layers=[2, 32, 32, 1])

# Fewer epochs
solver.train(n_epochs=2000)
```

### Better Accuracy
For higher accuracy:
```python
# Larger network
solver = HeatEquationPINN(layers=[2, 100, 100, 100, 100, 1])

# More training
solver.train(n_epochs=20000, n_pde=20000)
```

## Expected Outputs

All examples should produce:
- Training progress output
- Loss values decreasing over time
- Final error metrics (typically < 1% relative error)
- Visualization plots

## Troubleshooting

### "No module named 'torch'"
```bash
pip install torch numpy matplotlib
```

### "ModuleNotFoundError: No module named 'src'"
Make sure you're running from the correct directory or adjust the path:
```python
sys.path.append('/path/to/physicsinformedml/src')
```

### High error in results
- Train for more epochs
- Increase network size
- Use more collocation points
- Lower learning rate

## Contributing Examples

To contribute a new example:

1. Create a new file in this directory
2. Follow the naming convention: `description_demo.py`
3. Include docstring explaining what it demonstrates
4. Add entry to this README
5. Ensure it runs successfully
6. Submit a pull request

### Good Example Ideas
- Different thermal diffusivity values
- Custom initial conditions
- Parameter sensitivity analysis
- Comparison of network architectures
- Transfer learning demonstration
- Noise robustness testing

## Output Directory

Examples save their outputs to the `../outputs/` directory by default.

To specify a custom location:
```python
plot_solution(solver, save_path='/path/to/output.png')
```

## Performance Notes

Typical runtimes on CPU:
- Quick demo (2000 epochs): ~20-30 seconds
- Standard demo (5000 epochs): ~2-3 minutes
- High accuracy (20000 epochs): ~10-15 minutes

GPU acceleration (if available) can reduce these times by 5-10x.

---

Happy experimenting! 🚀
