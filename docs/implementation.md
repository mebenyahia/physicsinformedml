# Implementation Guide

This document provides detailed information about implementing and using the Physics-Informed Neural Network (PINN) code in this repository.

## Code Structure

### Main Components

1. **PINN Class** (`src/pinn_heat_equation.py`): Base neural network architecture
2. **HeatEquationPINN Class**: Specialized solver for the heat equation
3. **Visualization Functions**: Plotting and analysis tools
4. **Example Scripts**: Demonstration and usage examples

## PINN Class

### Initialization

```python
class PINN(nn.Module):
    def __init__(self, layers: list = [2, 50, 50, 50, 1]):
        """
        Args:
            layers: List defining network architecture
                   [input_dim, hidden1, hidden2, ..., output_dim]
        """
```

**Architecture Design:**
- Input layer: 2 neurons (x, t coordinates)
- Hidden layers: Typically 3-5 layers with 20-100 neurons each
- Output layer: 1 neuron (solution u)
- Activation: `tanh` (smooth, bounded, symmetric)

### Forward Pass

```python
def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute network output u(x, t)
    
    Args:
        x: Spatial coordinates [batch_size, 1]
        t: Temporal coordinates [batch_size, 1]
    
    Returns:
        Network prediction [batch_size, 1]
    """
```

**Implementation Details:**
1. Concatenate inputs: `[x, t]`
2. Apply layers sequentially with `tanh` activation
3. No activation on output layer

## HeatEquationPINN Class

### Initialization

```python
solver = HeatEquationPINN(
    alpha=0.1,                    # Thermal diffusivity
    layers=[2, 50, 50, 50, 1]     # Network architecture
)
```

**Parameters:**
- `alpha`: Thermal diffusivity constant (affects decay rate)
- `layers`: Network architecture specification

### PDE Loss Computation

```python
def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate PDE residual: ∂u/∂t - α∂²u/∂x²
    """
```

**Key Steps:**

1. **Enable Gradients:**
   ```python
   x.requires_grad = True
   t.requires_grad = True
   ```

2. **Forward Pass:**
   ```python
   u = self.model(x, t)
   ```

3. **First Derivatives:**
   ```python
   u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]
   u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]
   ```

4. **Second Derivative:**
   ```python
   u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                              create_graph=True)[0]
   ```

5. **Residual:**
   ```python
   pde_residual = u_t - self.alpha * u_xx
   return torch.mean(pde_residual ** 2)
   ```

### Training Function

```python
solver.train(
    n_epochs=10000,    # Number of training iterations
    n_pde=10000,       # Collocation points for PDE
    n_bc=200,          # Boundary condition points
    n_ic=200,          # Initial condition points
    lr=1e-3,           # Learning rate
    verbose=True       # Show progress bar
)
```

**Training Loop:**

1. **Generate Training Points:**
   - Random collocation points in domain
   - Points on boundaries
   - Points at initial time

2. **Compute Losses:**
   ```python
   loss_pde = self.pde_loss(x_pde, t_pde)
   loss_bc = self.boundary_loss(x_bc, t_bc, u_bc)
   loss_ic = self.initial_loss(x_ic, t_ic, u_ic)
   loss = loss_pde + loss_bc + loss_ic
   ```

3. **Backpropagation:**
   ```python
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   ```

4. **Record History:**
   ```python
   self.loss_history['total'].append(loss.item())
   ```

### Sampling Strategies

#### Collocation Points (PDE)

```python
def _generate_pde_points(self, n: int):
    """Random sampling in domain [0,1] × [0,1]"""
    x = torch.rand(n, 1, device=self.device)
    t = torch.rand(n, 1, device=self.device)
    return x, t
```

**Options:**
- Uniform random: Simple, works well
- Latin hypercube: Better coverage
- Adaptive: Focus on high-error regions

#### Boundary Points

```python
def _generate_bc_points(self, n: int):
    """Points on boundaries x=0 and x=1"""
    t = torch.rand(n, 1)
    x_left = torch.zeros(n//2, 1)   # x = 0
    x_right = torch.ones(n//2, 1)   # x = 1
    u_bc = torch.zeros(n, 1)        # Dirichlet BC
```

#### Initial Condition Points

```python
def _generate_ic_points(self, n: int):
    """Points at t=0 with initial condition"""
    x = torch.rand(n, 1)
    t = torch.zeros(n, 1)
    u = torch.sin(np.pi * x)  # u(x, 0) = sin(πx)
```

## Prediction and Evaluation

### Making Predictions

```python
# Single point
x = np.array([[0.5]])
t = np.array([[0.5]])
u_pred = solver.predict(x, t)

# Multiple points
x = np.linspace(0, 1, 100)[:, None]
t = np.linspace(0, 1, 100)[:, None]
X, T = np.meshgrid(x.flatten(), t.flatten())
u_pred = solver.predict(X.flatten()[:, None], T.flatten()[:, None])
```

### Computing Exact Solution

```python
u_exact = solver.exact_solution(x, t)
# u(x,t) = sin(πx) * exp(-π²αt)
```

### Error Metrics

```python
# L2 relative error
l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

# Maximum absolute error
max_error = np.max(np.abs(u_pred - u_exact))

# Mean absolute error
mae = np.mean(np.abs(u_pred - u_exact))
```

## Visualization

### Solution Plotting

```python
from pinn_heat_equation import plot_solution

fig = plot_solution(solver, save_path='outputs/solution.png')
```

**Generates:**
1. PINN solution heatmap
2. Exact solution heatmap
3. Absolute error map
4. Training loss history

### Custom Visualizations

```python
import matplotlib.pyplot as plt

# Time snapshots
time_steps = [0.0, 0.2, 0.5, 1.0]
x = np.linspace(0, 1, 200)[:, None]

for t_val in time_steps:
    t = np.ones_like(x) * t_val
    u = solver.predict(x, t)
    plt.plot(x, u, label=f't={t_val}')

plt.legend()
plt.show()
```

## Hyperparameter Tuning

### Network Architecture

**Width (neurons per layer):**
- Too few: Underfitting, can't represent solution
- Too many: Overfitting, slow training
- Typical range: 20-100 neurons

**Depth (number of layers):**
- Shallow (2-3): Simple problems
- Medium (4-6): Most problems
- Deep (7+): Very complex problems

**Rule of thumb:** Start with [2, 50, 50, 50, 1]

### Training Parameters

**Number of epochs:**
- Quick test: 1,000-2,000
- Standard: 5,000-10,000
- High accuracy: 20,000-50,000

**Learning rate:**
- Too high: Unstable, divergence
- Too low: Slow convergence
- Typical: 1e-3 to 1e-4
- Use learning rate scheduling for better results

**Number of collocation points:**
- More points: Better accuracy, slower training
- Typical: 5,000-10,000 for 1D problems
- Scale up for higher dimensions

### Loss Weights

For complex problems, weight the loss components:

```python
loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
```

**Balancing strategy:**
1. Start with equal weights (1.0 each)
2. If one component is much larger, reduce its weight
3. If convergence is slow, increase PDE weight

## Common Issues and Solutions

### Issue: High PDE Loss

**Symptoms:** PDE loss doesn't decrease, stays > 0.01

**Solutions:**
- Increase number of collocation points
- Increase network size
- Lower learning rate
- Train for more epochs

### Issue: Boundary/Initial Conditions Not Satisfied

**Symptoms:** BC/IC losses high, solution doesn't match boundaries

**Solutions:**
- Increase BC/IC point sampling
- Increase loss weights for BC/IC
- Check boundary condition implementation

### Issue: Slow Convergence

**Symptoms:** Training takes very long, loss decreases slowly

**Solutions:**
- Increase learning rate (carefully)
- Use learning rate scheduler
- Try different optimizer (Adam → L-BFGS)
- Reduce network size

### Issue: Unstable Training

**Symptoms:** Loss jumps around, NaN values

**Solutions:**
- Decrease learning rate
- Gradient clipping
- Better weight initialization
- Check for numerical issues in PDE formulation

## Advanced Techniques

### Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=500
)

for epoch in range(n_epochs):
    # ... training step ...
    scheduler.step(loss)
```

### Adaptive Sampling

Resample collocation points where error is high:

```python
# Compute residuals
with torch.no_grad():
    residuals = torch.abs(pde_residual)

# Sample more points where residual is high
probabilities = residuals / residuals.sum()
indices = torch.multinomial(probabilities, n_new_points, replacement=True)
```

### Transfer Learning

Pre-train on a similar problem:

```python
# Train on problem 1
solver1 = HeatEquationPINN(alpha=0.1)
solver1.train(...)

# Initialize solver 2 with solver 1's weights
solver2 = HeatEquationPINN(alpha=0.15)
solver2.model.load_state_dict(solver1.model.state_dict())
solver2.train(...)  # Fine-tune
```

## Performance Optimization

### GPU Acceleration

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
solver = HeatEquationPINN(alpha=0.1)
solver.model.to(device)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    with autocast():
        loss = compute_loss(...)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Vectorization

Compute all losses in parallel:

```python
# Instead of loop
for point in points:
    loss += compute_point_loss(point)

# Use vectorized operations
loss = torch.mean(compute_batch_loss(points))
```

## Testing and Validation

### Unit Tests

```python
def test_pde_residual():
    """Test that exact solution has zero residual"""
    solver = HeatEquationPINN(alpha=0.1)
    x = torch.tensor([[0.5]], requires_grad=True)
    t = torch.tensor([[0.0]], requires_grad=True)
    
    # At t=0, residual should be small
    residual = solver.pde_loss(x, t)
    assert residual < 1e-6
```

### Convergence Tests

```python
# Test different network sizes
for n_neurons in [20, 50, 100]:
    solver = HeatEquationPINN(layers=[2, n_neurons, n_neurons, 1])
    solver.train(...)
    error = compute_error(solver)
    print(f"Neurons: {n_neurons}, Error: {error}")
```

## Best Practices

1. **Start Simple**: Begin with small network, few epochs
2. **Monitor Training**: Watch all loss components
3. **Check Boundaries**: Verify BC/IC are satisfied
4. **Compare with Known Solutions**: Use exact solutions when available
5. **Validate Physically**: Ensure solution makes physical sense
6. **Save Checkpoints**: Save model during training
7. **Document Parameters**: Record all hyperparameters used

## References

For implementation details, see:
- PyTorch documentation: https://pytorch.org/docs/
- Automatic differentiation: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- Original PINN paper: Raissi et al. (2019)
