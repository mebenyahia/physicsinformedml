# Physics-Informed Machine Learning

A collection of hands-on resources that show how Physics-Informed Neural Networks (PINNs) blend deep learning with governing equations. The repository now contains two complementary learning tracks:

1. **Classic heat-equation PINN** – a compact PyTorch implementation that solves a 1D diffusion PDE with scripts and a notebook.
2. **Beginner-focused mass–spring PINN demo** – a self-contained walkthrough (code, notebooks, figures, and GitHub Pages site) living in `pinn-mass-spring-demo/`.

Both tracks emphasize readable code, extensive commentary, and reproducible experiments that run on a standard Python 3 environment.

## Overview

Physics-Informed Neural Networks (PINNs) are neural models that minimize measurement error *and* the residual of a known differential equation. Compared with purely data-driven approaches, PINNs:

- **Learn from sparse or noisy observations** by leveraging the physics as a regularizer.
- **Solve forward and inverse problems** without meshing complex domains.
- **Respect boundary/initial conditions** through explicit loss terms.
- **Generalize better** because the hypothesis space is constrained by the governing law.

This repository illustrates those ideas through two different problems described below.

## Included demos

### 1. Heat-equation PINN (original tutorial)
- Located in `src/pinn_heat_equation.py` with supporting material in `examples/` and `docs/`.
- Solves the 1D heat equation
  ```
  ∂u/∂t = α * ∂²u/∂x²
  ```
  with Dirichlet boundary conditions `u(0, t)=u(1, t)=0` and initial condition `u(x, 0)=sin(πx)`.
- Uses a fully connected network that takes `(x, t)` as input and outputs the temperature `u(x, t)`.
- Training minimizes a composite loss: PDE residual + boundary loss + initial-condition loss.
- Run it via:
  ```bash
  pip install -r requirements.txt
  python src/pinn_heat_equation.py            # full training script
  python examples/quick_demo.py               # lightweight illustration
  jupyter notebook examples/pinn_demo.ipynb   # interactive exploration
  ```

### 2. Mass–spring PINN demo (new beginner guide)
- Lives entirely inside `pinn-mass-spring-demo/` so it can be copied or pushed as its own repo.
- Focuses on the damped harmonic oscillator
  ```
  m * x''(t) + c * x'(t) + k * x(t) = 0
  ```
  with default parameters `m=1`, `k=1`, `c=0.1`.
- Contents:
  - `notebooks/01_mass_spring_physics_and_data.ipynb`: derives the physics, simulates clean/noisy data, and exports SVG figures.
  - `notebooks/02_pure_nn_vs_pinn.ipynb`: trains a baseline MLP and a PINN, compares their predictions, and summarizes takeaways.
  - `src/models.py`: reusable `TimeMLP` PyTorch module plus helper builders.
  - `figures/`: lightweight SVG plots referenced in the docs.
  - `docs/`: static GitHub Pages site with tabs for Home, Equation, Dataset, Models, Results, and Notebooks.
- To try it locally:
  ```bash
  cd pinn-mass-spring-demo
  python -m venv .venv && source .venv/bin/activate   # optional but recommended
  pip install -r requirements.txt
  jupyter notebook
  ```
  Then open the notebooks in numerical order. Enable GitHub Pages (source: `main`, folder: `/docs`) to publish the accompanying mini-site.

## How PINNs work (high level)

1. **Neural network architecture** – receives coordinates (time, or time+space) and outputs the field variable (displacement or temperature).
2. **Automatic differentiation** – PyTorch autograd computes the derivatives needed by each PDE/ODE.
3. **Loss composition** – combines data misfit terms with physics residuals and any boundary/initial-condition penalties.
4. **Training loop** – Adam or LBFGS optimizers update the network using batches of collocation points and observed samples.

## Repository structure

```
physicsinformedml/
├── README.md                     # You are here
├── requirements.txt              # Root dependency list (heat-equation demo)
├── setup.py                      # Package metadata (targets Python 3.8+)
├── src/
│   └── pinn_heat_equation.py     # Heat-equation PINN implementation
├── examples/
│   ├── quick_demo.py             # Minimal script
│   └── pinn_demo.ipynb           # Companion notebook
├── docs/                         # Theory/usage notes for the heat-equation tutorial
├── pinn-mass-spring-demo/        # Standalone mass–spring PINN walkthrough (code + docs)
│   ├── README.md                 # Beginner instructions specific to the demo
│   ├── requirements.txt          # Lightweight dependency list (PyTorch, SciPy, etc.)
│   ├── notebooks/                # Two guided Jupyter notebooks
│   ├── src/models.py             # Shared TimeMLP definition
│   └── docs/                     # GitHub Pages-ready HTML + CSS + SVG figures
└── ...                           # License, contributing guide, additional resources
```

## Contributing

We welcome issues and pull requests that improve clarity, fix bugs, or expand the set of demonstrators—just ensure new additions remain beginner friendly and stick to Python 3.8+ compatible syntax. See `CONTRIBUTING.md` for workflow details.

## License

This project is distributed under the terms of the MIT License (see `LICENSE`).
