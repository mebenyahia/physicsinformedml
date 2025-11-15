# Physics Informed Neural Networks for a Mass Spring System

## Overview
This repository walks you through a gentle, hands-on introduction to Physics Informed Neural Networks (PINNs) using one of the simplest mechanical systems: a mass attached to a spring. You will first build intuition about the physics, generate synthetic measurements, and then compare a regular neural network with a physics-aware PINN. Everything is explained in plain English and the code is carefully commented so that newcomers to deep learning feel comfortable.

A PINN is a neural network that knows about the governing physical law. Instead of only fitting the measurements, it also tries to satisfy the differential equation that describes the system. This extra guidance often leads to better predictions, especially when data are scarce or noisy.

## Problem Setup
We study the motion of a mass connected to a spring with optional damping. The displacement of the mass over time, noted as `x(t)`, follows the differential equation:

```
m * x''(t) + c * x'(t) + k * x(t) = 0
```

- `m` is the mass (we use 1 kg)
- `k` is the spring stiffness (we use 1 N/m)
- `c` is the damping coefficient (0.1 for a gently damped system)

The notebooks explain what each term means and show the resulting oscillatory motion.

## Repository Structure
```
pinn-mass-spring-demo/
├─ notebooks/          # Jupyter notebooks with explanations and experiments
├─ src/                # Reusable Python code (models for the neural networks)
├─ figures/            # Plots exported from the notebooks
├─ docs/               # Static GitHub Pages site with friendly explanations
├─ requirements.txt    # Python dependencies for the demo
└─ README.md           # You are here
```

## How to Run Locally
1. (Optional) Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
2. Install the Python packages.
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook.
   ```bash
   jupyter notebook
   ```
4. Open and run the notebooks in order:
   - `01_mass_spring_physics_and_data.ipynb`
   - `02_pure_nn_vs_pinn.ipynb`

## What You Will Learn
- Understand the basic physics of a mass-spring system and how damping changes the motion.
- See how a standard neural network behaves when trained on a small, noisy dataset.
- Learn how a Physics Informed Neural Network uses the differential equation to improve its predictions.
- Practice reading and running well-documented PyTorch code inside Jupyter notebooks.

## GitHub Pages Demo
A friendly static website in the `docs/` folder summarizes the project, explains the physics, and links to the notebooks. When this repository is hosted on GitHub, enable Pages for the `main` branch with the `docs/` folder as the source to explore the walkthrough in your browser.
