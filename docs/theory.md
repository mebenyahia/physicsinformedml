# Mathematical Theory of Physics-Informed Neural Networks

## Introduction

Physics-Informed Neural Networks (PINNs) provide a novel approach to solving partial differential equations (PDEs) by incorporating physical laws directly into the neural network training process. This document explains the mathematical foundations of PINNs.

## Problem Formulation

### General PDE

Consider a general PDE of the form:

```
f(u, ∂u/∂t, ∂u/∂x, ∂²u/∂x², ...; x, t) = 0
```

where:
- `u(x, t)` is the unknown solution
- `x` represents spatial coordinates
- `t` represents time
- `f` is a differential operator

### Boundary and Initial Conditions

The PDE is supplemented with:

**Boundary Conditions (BC):**
```
B(u; x, t) = 0,  for (x, t) ∈ ∂Ω
```

**Initial Conditions (IC):**
```
u(x, 0) = u₀(x),  for x ∈ Ω
```

where:
- `Ω` is the spatial domain
- `∂Ω` is the boundary of the domain
- `u₀(x)` is the initial state

## Neural Network Approximation

### Network Architecture

A PINN approximates the solution using a neural network:

```
u(x, t) ≈ u_θ(x, t) = NN(x, t; θ)
```

where:
- `u_θ` is the neural network approximation
- `θ` represents all trainable parameters (weights and biases)
- `NN` is a fully connected feed-forward neural network

### Standard Architecture

For a network with L layers:

```
Input: (x, t)
Hidden Layer 1: h₁ = σ(W₁[x, t]ᵀ + b₁)
Hidden Layer 2: h₂ = σ(W₂h₁ + b₂)
...
Hidden Layer L-1: hₗ₋₁ = σ(Wₗ₋₁hₗ₋₂ + bₗ₋₁)
Output: u = Wₗhₗ₋₁ + bₗ
```

where:
- `σ` is the activation function (typically tanh or sigmoid)
- `Wᵢ` and `bᵢ` are weights and biases of layer i
- `θ = {W₁, b₁, W₂, b₂, ..., Wₗ, bₗ}`

## Loss Function

The key innovation of PINNs is the composite loss function that enforces both data fitting and physics compliance.

### Complete Loss Function

```
L(θ) = λ_PDE · L_PDE(θ) + λ_BC · L_BC(θ) + λ_IC · L_IC(θ) + λ_data · L_data(θ)
```

where:
- `λ_PDE, λ_BC, λ_IC, λ_data` are weight coefficients
- Each component measures different aspects of the solution

### 1. PDE Loss (Physics Loss)

The PDE loss enforces that the network solution satisfies the governing equation:

```
L_PDE(θ) = (1/N_f) Σᵢ₌₁ᴺᶠ |f(u_θ, ∂u_θ/∂t, ∂u_θ/∂x, ...; xᵢ, tᵢ)|²
```

where:
- `N_f` is the number of collocation points
- `(xᵢ, tᵢ)` are randomly sampled points in the domain
- The PDE residual `f(...)` should be zero everywhere

### 2. Boundary Condition Loss

The BC loss ensures the solution satisfies boundary conditions:

```
L_BC(θ) = (1/N_b) Σᵢ₌₁ᴺᵇ |u_θ(xᵢ, tᵢ) - u_b(xᵢ, tᵢ)|²
```

where:
- `N_b` is the number of boundary points
- `(xᵢ, tᵢ) ∈ ∂Ω` are points on the boundary
- `u_b` is the prescribed boundary value

### 3. Initial Condition Loss

The IC loss enforces initial conditions:

```
L_IC(θ) = (1/N_i) Σᵢ₌₁ᴺⁱ |u_θ(xᵢ, 0) - u₀(xᵢ)|²
```

where:
- `N_i` is the number of initial condition points
- `u₀(xᵢ)` is the prescribed initial state

### 4. Data Loss (Optional)

If measurement data is available:

```
L_data(θ) = (1/N_d) Σᵢ₌₁ᴺᵈ |u_θ(xᵢ, tᵢ) - uᵢᵈᵃᵗᵃ|²
```

## Automatic Differentiation

### Computing Derivatives

One of the key advantages of PINNs is the use of automatic differentiation to compute derivatives efficiently.

For a function `u_θ(x, t)`:

**First-order derivatives:**
```
∂u_θ/∂x = ∂/∂x[NN(x, t; θ)]
∂u_θ/∂t = ∂/∂t[NN(x, t; θ)]
```

**Second-order derivatives:**
```
∂²u_θ/∂x² = ∂/∂x[∂u_θ/∂x]
```

### Chain Rule Application

Modern deep learning frameworks (PyTorch, TensorFlow) implement automatic differentiation using the chain rule:

```
∂u/∂x = (∂u/∂hₗ₋₁) · (∂hₗ₋₁/∂hₗ₋₂) · ... · (∂h₁/∂x)
```

## Training Algorithm

### Optimization Problem

Find optimal parameters:

```
θ* = argmin_θ L(θ)
```

### Gradient Descent

Update parameters iteratively:

```
θᵏ⁺¹ = θᵏ - η∇_θL(θᵏ)
```

where:
- `η` is the learning rate
- `∇_θL` is the gradient of the loss with respect to parameters

### Common Optimizers

1. **Adam (Adaptive Moment Estimation)**
   - Combines momentum and adaptive learning rates
   - Generally performs well for PINNs

2. **L-BFGS (Limited-memory BFGS)**
   - Quasi-Newton method
   - Can provide faster convergence for smooth problems

## Example: 1D Heat Equation

### Problem Statement

```
∂u/∂t = α∂²u/∂x²,  x ∈ [0, 1], t ∈ [0, T]
```

Boundary conditions:
```
u(0, t) = 0
u(1, t) = 0
```

Initial condition:
```
u(x, 0) = sin(πx)
```

### Analytical Solution

```
u(x, t) = sin(πx)e^(-π²αt)
```

### PINN Loss Components

**PDE Residual:**
```
f = ∂u_θ/∂t - α∂²u_θ/∂x²
```

**Loss Function:**
```
L = (1/N_f)Σ|∂u_θ/∂t - α∂²u_θ/∂x²|² +
    (1/N_b)Σ|u_θ(0,t)|² + |u_θ(1,t)|² +
    (1/N_i)Σ|u_θ(x,0) - sin(πx)|²
```

## Convergence and Approximation Theory

### Universal Approximation Theorem

Neural networks with sufficient width can approximate any continuous function to arbitrary accuracy.

### Convergence Conditions

For PINNs to converge to the true solution:

1. **Network Expressiveness**: The network must be capable of representing the solution
2. **Sufficient Sampling**: Adequate collocation and boundary points
3. **Proper Training**: Optimization must find a good minimum
4. **Well-Posed Problem**: The underlying PDE must have a unique solution

### Error Analysis

The approximation error consists of:

1. **Approximation Error**: How well can the network represent the true solution
2. **Optimization Error**: How close are we to the optimal parameters
3. **Generalization Error**: Performance on unseen points

## Advanced Topics

### Multi-Scale Problems

For problems with multiple scales, consider:
- **Adaptive activation functions**
- **Fourier features**
- **Multi-fidelity approaches**

### Inverse Problems

PINNs naturally handle inverse problems where we want to identify unknown parameters:

```
L = L_PDE(θ, λ) + L_data(θ)
```

where `λ` represents unknown physical parameters (e.g., diffusivity, viscosity).

### Uncertainty Quantification

Bayesian PINNs can quantify uncertainty:

```
p(θ|data) ∝ p(data|θ)p(θ)
```

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). Physics informed deep learning (part I): Data-driven solutions of nonlinear partial differential equations. *arXiv preprint arXiv:1711.10561*.

3. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422-440.
