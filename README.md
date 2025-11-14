# Physics-Informed Machine Learning

A comprehensive demonstration of Physics-Informed Neural Networks (PINNs) for solving partial differential equations.

## Overview

This repository provides a complete implementation and tutorial on Physics-Informed Neural Networks (PINNs), showcasing how deep learning can be combined with physics to solve complex differential equations without traditional numerical methods.

### What are Physics-Informed Neural Networks?

Physics-Informed Neural Networks (PINNs) are a class of deep learning models that incorporate physical laws, expressed as partial differential equations (PDEs), directly into the neural network training process. Unlike conventional machine learning approaches that rely solely on data, PINNs leverage known physics to:

- **Learn from sparse and noisy data**
- **Solve forward and inverse problems** in differential equations
- **Provide mesh-free solutions** to complex PDEs
- **Respect physical constraints** automatically
- **Generalize better** with limited training data

## Features

✨ **Complete PINN Implementation**: Fully documented PyTorch implementation of PINNs for solving the 1D heat equation

📊 **Interactive Jupyter Notebook**: Step-by-step tutorial with visualizations and explanations

🎯 **Comprehensive Examples**: Multiple examples demonstrating different aspects of PINNs

📚 **Detailed Documentation**: In-depth explanations of theory, implementation, and usage

🔬 **Visualization Tools**: Built-in plotting functions to visualize solutions and errors

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mebenyahia/physicsinformedml.git
cd physicsinformedml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Demo

**Python Script:**
```bash
python src/pinn_heat_equation.py
```

**Quick Demo:**
```bash
python examples/quick_demo.py
```

**Jupyter Notebook:**
```bash
jupyter notebook examples/pinn_demo.ipynb
```

## The Problem: 1D Heat Equation

The demo solves the 1D heat equation, which describes how heat diffuses through a material over time:

```
∂u/∂t = α * ∂²u/∂x²
```

Where:
- `u(x, t)` is the temperature at position `x` and time `t`
- `α` is the thermal diffusivity constant

### Boundary Conditions
- `u(0, t) = 0` (left boundary fixed at 0)
- `u(1, t) = 0` (right boundary fixed at 0)

### Initial Condition
- `u(x, 0) = sin(πx)` (initial temperature distribution)

### Exact Solution
For validation, the exact solution is:
```
u(x, t) = sin(πx) * exp(-π²αt)
```

## How PINNs Work

### 1. Neural Network Architecture

The PINN takes spatial and temporal coordinates `(x, t)` as input and outputs the solution `u(x, t)`:

```
Input: (x, t) → Neural Network → Output: u(x, t)
```

### 2. Loss Function Components

The PINN is trained to minimize a composite loss function:

**L_total = L_PDE + L_BC + L_IC**

Where:
- **L_PDE**: PDE residual loss - ensures the solution satisfies the heat equation
- **L_BC**: Boundary condition loss - ensures `u(0, t) = 0` and `u(1, t) = 0`
- **L_IC**: Initial condition loss - ensures `u(x, 0) = sin(πx)`

### 3. Automatic Differentiation

PyTorch's autograd is used to compute the derivatives needed for the PDE:
```python
u_t = ∂u/∂t   # First derivative w.r.t. time
u_x = ∂u/∂x   # First derivative w.r.t. space
u_xx = ∂²u/∂x² # Second derivative w.r.t. space
```

### 4. Training Process

1. Sample random collocation points in the domain
2. Sample points on boundaries and initial condition
3. Compute network predictions
4. Calculate derivatives using autograd
5. Compute loss function
6. Update network weights using gradient descent

## Repository Structure

```
physicsinformedml/
├── src/
│   └── pinn_heat_equation.py    # Main PINN implementation
├── examples/
│   ├── quick_demo.py            # Quick demonstration script
│   └── pinn_demo.ipynb          # Interactive Jupyter notebook
├── docs/
│   ├── theory.md                # Mathematical theory
│   ├── implementation.md        # Implementation details
│   └── usage.md                 # Usage guide
├── outputs/                     # Generated plots and results
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## Code Example

```python
from src.pinn_heat_equation import HeatEquationPINN, plot_solution
import torch

# Initialize PINN
solver = HeatEquationPINN(alpha=0.1, layers=[2, 50, 50, 50, 1])

# Train the network
solver.train(n_epochs=5000, n_pde=10000, verbose=True)

# Visualize results
plot_solution(solver, save_path='outputs/solution.png')

# Make predictions
import numpy as np
x = np.array([[0.5]])  # x = 0.5
t = np.array([[0.5]])  # t = 0.5
u = solver.predict(x, t)
print(f"u(0.5, 0.5) = {u[0, 0]:.6f}")
```

## Results

The PINN achieves excellent accuracy in solving the heat equation:
- **Relative L2 Error**: < 0.001
- **Maximum Absolute Error**: < 0.01
- **Training Time**: ~2-5 minutes on CPU

Example output showing the PINN solution, exact solution, error, and training history:

![Solution Visualization](outputs/heat_equation_solution.png)

## Key Advantages of PINNs

1. **No Mesh Required**: Unlike finite difference or finite element methods, PINNs don't require discretization of the domain

2. **Handles Complex Geometries**: Can solve PDEs in irregular domains without complex meshing

3. **Data-Driven Physics**: Can incorporate both data and physical laws, useful when physics is partially known

4. **Inverse Problems**: Can solve inverse problems (finding unknown parameters) naturally

5. **Continuous Solution**: Provides a continuous function that can be evaluated anywhere in the domain

6. **Automatic Differentiation**: Leverages modern deep learning frameworks for efficient derivative computation

## Limitations and Considerations

- **Computational Cost**: Training can be expensive for complex problems
- **Hyperparameter Sensitivity**: Performance depends on network architecture and training parameters
- **Convergence**: May require careful initialization and tuning
- **High-Dimensional Problems**: Can struggle with very high-dimensional PDEs

## Extensions and Advanced Topics

### Other PDEs to Try
- **Wave Equation**: Oscillatory phenomena
- **Burgers' Equation**: Fluid dynamics with nonlinearity
- **Navier-Stokes**: Fluid flow (more complex)
- **Schrödinger Equation**: Quantum mechanics

### Advanced Techniques
- **Transfer Learning**: Pre-train on similar problems
- **Multi-fidelity**: Combine low and high-fidelity data
- **Adaptive Sampling**: Focus collocation points where error is high
- **Ensemble Methods**: Multiple PINNs for uncertainty quantification

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add new PDE examples
- Improve documentation

## References

### Foundational Papers

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.
   - Original PINN paper introducing the methodology

2. **Karniadakis, G. E., et al.** (2021). *Physics-informed machine learning*. Nature Reviews Physics, 3(6), 422-440.
   - Comprehensive review of physics-informed ML

3. **Cuomo, S., et al.** (2022). *Scientific machine learning through physics–informed neural networks: Where we are and what's next*. Journal of Scientific Computing, 92(3), 88.
   - Recent survey of PINN applications and future directions

### Additional Resources

- [PINN Papers on GitHub](https://github.com/maziarraissi/PINNs)
- [DeepXDE Library](https://deepxde.readthedocs.io/)
- [SciML Ecosystem](https://sciml.ai/)

## License

This project is provided for educational purposes. Feel free to use and modify the code for learning and research.

## Acknowledgments

This implementation is inspired by the pioneering work of Maziar Raissi, Paris Perdikaris, and George Em Karniadakis on Physics-Informed Neural Networks.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Learning! 🚀🔬**