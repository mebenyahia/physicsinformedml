# Usage Guide

This guide provides practical instructions for using the Physics-Informed Neural Network implementation.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mebenyahia/physicsinformedml.git
   cd physicsinformedml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
   ```

### Optional: Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running the Main Demo

```bash
python src/pinn_heat_equation.py
```

**Expected output:**
- Training progress with loss values
- Error metrics (L2 error, max error)
- Generated plot saved to `outputs/heat_equation_solution.png`

**Typical runtime:** 2-5 minutes on CPU

### Running Quick Demo

```bash
python examples/quick_demo.py
```

This runs a faster version with fewer epochs for quick testing.

### Running Jupyter Notebook

```bash
jupyter notebook examples/pinn_demo.ipynb
```

This opens an interactive notebook with step-by-step explanations.

## Basic Usage

### Example 1: Simple Training

```python
from src.pinn_heat_equation import HeatEquationPINN
import torch
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Create solver
solver = HeatEquationPINN(alpha=0.1)

# Train
solver.train(n_epochs=5000, verbose=True)

# Make prediction
x = np.array([[0.5]])
t = np.array([[0.5]])
u = solver.predict(x, t)

print(f"u(0.5, 0.5) = {u[0, 0]:.6f}")
```

### Example 2: Custom Network Architecture

```python
# Smaller network for faster training
solver = HeatEquationPINN(
    alpha=0.1,
    layers=[2, 32, 32, 1]  # 2 hidden layers with 32 neurons each
)

# Larger network for better accuracy
solver = HeatEquationPINN(
    alpha=0.1,
    layers=[2, 100, 100, 100, 100, 1]  # 4 hidden layers with 100 neurons
)
```

### Example 3: Adjusting Training Parameters

```python
solver = HeatEquationPINN(alpha=0.1)

# More epochs for higher accuracy
solver.train(
    n_epochs=20000,      # More training iterations
    n_pde=15000,         # More collocation points
    n_bc=400,            # More boundary points
    n_ic=400,            # More initial condition points
    lr=5e-4,             # Lower learning rate
    verbose=True
)
```

### Example 4: Different Thermal Diffusivity

```python
# Lower diffusivity (slower heat diffusion)
solver1 = HeatEquationPINN(alpha=0.05)
solver1.train(n_epochs=5000)

# Higher diffusivity (faster heat diffusion)
solver2 = HeatEquationPINN(alpha=0.2)
solver2.train(n_epochs=5000)
```

## Visualization Examples

### Example 5: Plot Solution

```python
from src.pinn_heat_equation import plot_solution
import matplotlib.pyplot as plt

solver = HeatEquationPINN(alpha=0.1)
solver.train(n_epochs=5000)

# Generate and show plot
fig = plot_solution(solver)
plt.show()

# Or save to file
plot_solution(solver, save_path='my_solution.png')
```

### Example 6: Custom Time Snapshots

```python
import matplotlib.pyplot as plt
import numpy as np

solver = HeatEquationPINN(alpha=0.1)
solver.train(n_epochs=5000)

# Plot at different times
times = [0.0, 0.1, 0.3, 0.5, 1.0]
x = np.linspace(0, 1, 200)[:, None]

plt.figure(figsize=(12, 6))
for t_val in times:
    t = np.ones_like(x) * t_val
    u_pred = solver.predict(x, t)
    u_exact = solver.exact_solution(x, t)
    
    plt.plot(x, u_pred, label=f't={t_val:.1f} (PINN)', linewidth=2)
    plt.plot(x, u_exact, '--', alpha=0.5)

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Solution Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('time_evolution.png', dpi=150)
plt.show()
```

### Example 7: Error Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

solver = HeatEquationPINN(alpha=0.1)
solver.train(n_epochs=5000)

# Create test grid
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t)

# Compute solutions
X_flat = X.flatten()[:, None]
T_flat = T.flatten()[:, None]
U_pred = solver.predict(X_flat, T_flat).reshape(X.shape)
U_exact = solver.exact_solution(X_flat, T_flat).reshape(X.shape)

# Plot error
error = np.abs(U_pred - U_exact)
plt.figure(figsize=(10, 8))
plt.contourf(X, T, error, levels=50, cmap='hot')
plt.colorbar(label='Absolute Error')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Error Distribution')
plt.savefig('error_map.png', dpi=150)
plt.show()

# Error statistics
print(f"Mean error: {error.mean():.6f}")
print(f"Std error: {error.std():.6f}")
print(f"Max error: {error.max():.6f}")
```

## Advanced Usage

### Example 8: Monitor Training Progress

```python
solver = HeatEquationPINN(alpha=0.1)

# Train and get loss history
loss_history = solver.train(n_epochs=10000, verbose=True)

# Plot detailed loss history
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total loss
axes[0, 0].plot(loss_history['total'])
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('Total Loss')
axes[0, 0].grid(True)

# PDE loss
axes[0, 1].plot(loss_history['pde'])
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('PDE Loss')
axes[0, 1].grid(True)

# BC loss
axes[1, 0].plot(loss_history['bc'])
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('Boundary Condition Loss')
axes[1, 0].grid(True)

# IC loss
axes[1, 1].plot(loss_history['ic'])
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Initial Condition Loss')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('loss_components.png', dpi=150)
plt.show()
```

### Example 9: Convergence Study

```python
import numpy as np
import matplotlib.pyplot as plt

# Test different network sizes
neuron_counts = [20, 30, 50, 70, 100]
errors = []

for n in neuron_counts:
    print(f"\nTraining with {n} neurons per layer...")
    solver = HeatEquationPINN(
        alpha=0.1,
        layers=[2, n, n, n, 1]
    )
    solver.train(n_epochs=5000, verbose=False)
    
    # Compute error
    x = np.linspace(0, 1, 100)[:, None]
    t = np.linspace(0, 1, 100)[:, None]
    X, T = np.meshgrid(x.flatten(), t.flatten())
    U_pred = solver.predict(X.flatten()[:, None], T.flatten()[:, None])
    U_exact = solver.exact_solution(X.flatten()[:, None], T.flatten()[:, None])
    
    l2_error = np.linalg.norm(U_pred - U_exact) / np.linalg.norm(U_exact)
    errors.append(l2_error)
    print(f"L2 Error: {l2_error:.6f}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(neuron_counts, errors, 'o-', linewidth=2, markersize=8)
plt.xlabel('Neurons per Layer')
plt.ylabel('Relative L2 Error')
plt.title('Convergence vs Network Size')
plt.grid(True, alpha=0.3)
plt.savefig('convergence_study.png', dpi=150)
plt.show()
```

### Example 10: Save and Load Model

```python
import torch

# Train and save
solver = HeatEquationPINN(alpha=0.1)
solver.train(n_epochs=5000)
torch.save(solver.model.state_dict(), 'pinn_model.pth')
print("Model saved!")

# Load later
solver_loaded = HeatEquationPINN(alpha=0.1)
solver_loaded.model.load_state_dict(torch.load('pinn_model.pth'))
solver_loaded.model.eval()
print("Model loaded!")

# Use loaded model
x = np.array([[0.5]])
t = np.array([[0.5]])
u = solver_loaded.predict(x, t)
print(f"Prediction: {u[0, 0]:.6f}")
```

## Troubleshooting

### Issue: Import Error

**Problem:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install torch numpy matplotlib
```

### Issue: CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size (n_pde, n_bc, n_ic)
- Use smaller network
- Use CPU instead: Set `CUDA_VISIBLE_DEVICES=""`

### Issue: Poor Accuracy

**Problem:** High error, solution doesn't match exact solution

**Solutions:**
1. Train longer:
   ```python
   solver.train(n_epochs=20000)
   ```

2. Use more collocation points:
   ```python
   solver.train(n_pde=20000, n_bc=500, n_ic=500)
   ```

3. Increase network size:
   ```python
   solver = HeatEquationPINN(layers=[2, 100, 100, 100, 1])
   ```

4. Lower learning rate:
   ```python
   solver.train(lr=5e-4)
   ```

### Issue: Training Unstable

**Problem:** Loss increases or becomes NaN

**Solutions:**
1. Lower learning rate:
   ```python
   solver.train(lr=1e-4)
   ```

2. Gradient clipping:
   ```python
   # Modify training loop to add:
   torch.nn.utils.clip_grad_norm_(solver.model.parameters(), max_norm=1.0)
   ```

### Issue: Slow Training

**Problem:** Training takes too long

**Solutions:**
1. Use GPU if available
2. Reduce network size
3. Reduce number of epochs
4. Use fewer collocation points

## Tips and Best Practices

### 1. Start Small
Begin with a small network and few epochs to verify everything works:
```python
solver = HeatEquationPINN(layers=[2, 20, 20, 1])
solver.train(n_epochs=1000)
```

### 2. Monitor All Losses
Check that all loss components are decreasing:
```python
loss_history = solver.train(...)
plt.plot(loss_history['pde'], label='PDE')
plt.plot(loss_history['bc'], label='BC')
plt.plot(loss_history['ic'], label='IC')
plt.legend()
plt.yscale('log')
plt.show()
```

### 3. Verify Boundary Conditions
Check that boundary conditions are satisfied:
```python
# Test left boundary (x=0)
x = np.zeros((10, 1))
t = np.random.rand(10, 1)
u = solver.predict(x, t)
print(f"Left boundary error: {np.abs(u).max():.6f}")  # Should be ~0
```

### 4. Use Version Control
Save different configurations:
```python
# Save configuration
config = {
    'alpha': 0.1,
    'layers': [2, 50, 50, 50, 1],
    'n_epochs': 5000,
    'lr': 1e-3
}
torch.save({
    'model': solver.model.state_dict(),
    'config': config,
    'loss_history': solver.loss_history
}, 'checkpoint.pth')
```

### 5. Compare with Exact Solution
Always validate against known solutions when available:
```python
x_test = np.random.rand(100, 1)
t_test = np.random.rand(100, 1)
u_pred = solver.predict(x_test, t_test)
u_exact = solver.exact_solution(x_test, t_test)
error = np.abs(u_pred - u_exact)
print(f"Mean error: {error.mean():.6f}")
```

## Performance Considerations

### CPU vs GPU

**CPU (Default):**
- Works everywhere
- Slower for large networks
- Fine for this demo

**GPU:**
- Much faster
- Requires CUDA-capable GPU
- Automatically used if available

### Memory Usage

For large problems, consider:
- Batch processing predictions
- Reducing network size
- Using gradient checkpointing

### Speed Optimization

```python
# Disable unnecessary gradient computation
with torch.no_grad():
    u = solver.predict(x, t)

# Use float32 (default) instead of float64
# Already using float32 in implementation
```

## Next Steps

1. **Explore the Jupyter Notebook**: `examples/pinn_demo.ipynb`
2. **Read the Theory**: `docs/theory.md`
3. **Study Implementation**: `docs/implementation.md`
4. **Modify Parameters**: Experiment with different settings
5. **Try Other Problems**: Implement wave equation, Burgers' equation, etc.

## Getting Help

- Check documentation in `docs/` directory
- Review examples in `examples/` directory
- Read code comments in `src/pinn_heat_equation.py`
- Open an issue on GitHub for bugs or questions

## Additional Resources

- PyTorch tutorials: https://pytorch.org/tutorials/
- PINN papers: See References in README.md
- Deep learning basics: https://www.deeplearningbook.org/

---

Happy experimenting with Physics-Informed Neural Networks! 🚀
