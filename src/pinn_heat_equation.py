"""
Physics-Informed Neural Network (PINN) Implementation

This module implements a PINN for solving partial differential equations.
The example demonstrates solving the 1D heat equation:
    ∂u/∂t = α * ∂²u/∂x²
where u(x,t) is the temperature, α is the thermal diffusivity.

Boundary conditions:
    u(0, t) = 0
    u(1, t) = 0
    u(x, 0) = sin(πx)

Exact solution: u(x, t) = sin(πx) * exp(-π²αt)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.
    
    The network learns to satisfy both the PDE and boundary/initial conditions
    by incorporating physics-based loss terms.
    """
    
    def __init__(self, layers: list = [2, 50, 50, 50, 1]):
        """
        Initialize the PINN.
        
        Args:
            layers: List of integers defining the network architecture.
                   First element is input dimension (x, t), last is output (u).
        """
        super(PINN, self).__init__()
        
        self.layers_list = layers
        self.network = nn.ModuleList()
        
        # Build the neural network
        for i in range(len(layers) - 1):
            self.network.append(nn.Linear(layers[i], layers[i+1]))
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Spatial coordinate(s)
            t: Temporal coordinate(s)
            
        Returns:
            Network prediction u(x, t)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        
        # Pass through network with tanh activation
        for i, layer in enumerate(self.network[:-1]):
            inputs = torch.tanh(layer(inputs))
        
        # Output layer (no activation)
        output = self.network[-1](inputs)
        
        return output


class HeatEquationPINN:
    """
    PINN solver for the 1D heat equation.
    
    Solves: ∂u/∂t = α * ∂²u/∂x²
    """
    
    def __init__(self, alpha: float = 0.1, layers: list = [2, 50, 50, 50, 1]):
        """
        Initialize the heat equation PINN solver.
        
        Args:
            alpha: Thermal diffusivity constant
            layers: Network architecture
        """
        self.alpha = alpha
        self.model = PINN(layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss history
        self.loss_history = {
            'total': [],
            'pde': [],
            'bc': [],
            'ic': []
        }
    
    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate the PDE residual loss.
        
        The PDE is: ∂u/∂t - α * ∂²u/∂x² = 0
        """
        x.requires_grad = True
        t.requires_grad = True
        
        u = self.model(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]
        
        # PDE residual
        pde_residual = u_t - self.alpha * u_xx
        
        return torch.mean(pde_residual ** 2)
    
    def boundary_loss(self, x_bc: torch.Tensor, t_bc: torch.Tensor,
                      u_bc: torch.Tensor) -> torch.Tensor:
        """Calculate boundary condition loss."""
        u_pred = self.model(x_bc, t_bc)
        return torch.mean((u_pred - u_bc) ** 2)
    
    def initial_loss(self, x_ic: torch.Tensor, t_ic: torch.Tensor,
                     u_ic: torch.Tensor) -> torch.Tensor:
        """Calculate initial condition loss."""
        u_pred = self.model(x_ic, t_ic)
        return torch.mean((u_pred - u_ic) ** 2)
    
    def train(self, n_epochs: int = 10000, n_pde: int = 10000,
              n_bc: int = 200, n_ic: int = 200, lr: float = 1e-3,
              verbose: bool = True) -> dict:
        """
        Train the PINN.
        
        Args:
            n_epochs: Number of training epochs
            n_pde: Number of collocation points for PDE
            n_bc: Number of boundary condition points
            n_ic: Number of initial condition points
            lr: Learning rate
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary containing loss history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Generate training data
        x_pde, t_pde = self._generate_pde_points(n_pde)
        x_bc, t_bc, u_bc = self._generate_bc_points(n_bc)
        x_ic, t_ic, u_ic = self._generate_ic_points(n_ic)
        
        # Training loop
        iterator = tqdm(range(n_epochs)) if verbose else range(n_epochs)
        
        for epoch in iterator:
            optimizer.zero_grad()
            
            # Calculate losses
            loss_pde = self.pde_loss(x_pde, t_pde)
            loss_bc = self.boundary_loss(x_bc, t_bc, u_bc)
            loss_ic = self.initial_loss(x_ic, t_ic, u_ic)
            
            # Total loss
            loss = loss_pde + loss_bc + loss_ic
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record losses
            self.loss_history['total'].append(loss.item())
            self.loss_history['pde'].append(loss_pde.item())
            self.loss_history['bc'].append(loss_bc.item())
            self.loss_history['ic'].append(loss_ic.item())
            
            # Update progress bar
            if verbose and epoch % 100 == 0:
                iterator.set_description(
                    f"Loss: {loss.item():.6f} | PDE: {loss_pde.item():.6f} | "
                    f"BC: {loss_bc.item():.6f} | IC: {loss_ic.item():.6f}"
                )
        
        return self.loss_history
    
    def _generate_pde_points(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random collocation points in the domain."""
        x = torch.rand(n, 1, device=self.device)
        t = torch.rand(n, 1, device=self.device)
        return x, t
    
    def _generate_bc_points(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate boundary condition points."""
        t = torch.rand(n, 1, device=self.device)
        
        # Left boundary: x = 0
        x_left = torch.zeros(n // 2, 1, device=self.device)
        t_left = t[:n // 2]
        u_left = torch.zeros(n // 2, 1, device=self.device)
        
        # Right boundary: x = 1
        x_right = torch.ones(n // 2, 1, device=self.device)
        t_right = t[n // 2:]
        u_right = torch.zeros(n // 2, 1, device=self.device)
        
        # Combine
        x_bc = torch.cat([x_left, x_right], dim=0)
        t_bc = torch.cat([t_left, t_right], dim=0)
        u_bc = torch.cat([u_left, u_right], dim=0)
        
        return x_bc, t_bc, u_bc
    
    def _generate_ic_points(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate initial condition points."""
        x = torch.rand(n, 1, device=self.device)
        t = torch.zeros(n, 1, device=self.device)
        u = torch.sin(np.pi * x)  # Initial condition: u(x, 0) = sin(πx)
        
        return x, t, u
    
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict solution at given points.
        
        Args:
            x: Spatial coordinates (numpy array)
            t: Temporal coordinates (numpy array)
            
        Returns:
            Predicted solution u(x, t)
        """
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            u_pred = self.model(x_tensor, t_tensor)
            
        return u_pred.cpu().numpy()
    
    def exact_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Calculate the exact solution for comparison.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            Exact solution u(x, t) = sin(πx) * exp(-π²αt)
        """
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * self.alpha * t)


def plot_solution(solver: HeatEquationPINN, save_path: Optional[str] = None):
    """
    Visualize the PINN solution and compare with exact solution.
    
    Args:
        solver: Trained HeatEquationPINN solver
        save_path: Optional path to save the figure
    """
    # Create mesh
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    
    # Predict
    U_pred = solver.predict(X_flat, T_flat).reshape(X.shape)
    U_exact = solver.exact_solution(X_flat, T_flat).reshape(X.shape)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PINN solution
    im1 = axes[0, 0].contourf(X, T, U_pred, levels=50, cmap='hot')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    axes[0, 0].set_title('PINN Solution')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Exact solution
    im2 = axes[0, 1].contourf(X, T, U_exact, levels=50, cmap='hot')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    axes[0, 1].set_title('Exact Solution')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Absolute error
    error = np.abs(U_pred - U_exact)
    im3 = axes[1, 0].contourf(X, T, error, levels=50, cmap='viridis')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    axes[1, 0].set_title(f'Absolute Error (Max: {error.max():.6f})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Loss history
    axes[1, 1].semilogy(solver.loss_history['total'], label='Total Loss')
    axes[1, 1].semilogy(solver.loss_history['pde'], label='PDE Loss', alpha=0.7)
    axes[1, 1].semilogy(solver.loss_history['bc'], label='BC Loss', alpha=0.7)
    axes[1, 1].semilogy(solver.loss_history['ic'], label='IC Loss', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Loss History')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def main():
    """Main function to demonstrate PINN training and visualization."""
    print("=" * 70)
    print("Physics-Informed Neural Network Demo")
    print("Solving 1D Heat Equation: ∂u/∂t = α * ∂²u/∂x²")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize solver
    print("Initializing PINN solver...")
    alpha = 0.1  # Thermal diffusivity
    solver = HeatEquationPINN(alpha=alpha, layers=[2, 50, 50, 50, 1])
    
    print(f"Device: {solver.device}")
    print(f"Thermal diffusivity (α): {alpha}")
    print(f"Network architecture: {solver.model.layers_list}")
    print()
    
    # Train
    print("Training PINN...")
    print("-" * 70)
    solver.train(n_epochs=5000, n_pde=10000, n_bc=200, n_ic=200,
                 lr=1e-3, verbose=True)
    print()
    print("Training completed!")
    print()
    
    # Calculate error metrics
    print("Calculating error metrics...")
    x_test = np.linspace(0, 1, 100)[:, None]
    t_test = np.linspace(0, 1, 100)[:, None]
    X_test, T_test = np.meshgrid(x_test.flatten(), t_test.flatten())
    X_flat = X_test.flatten()[:, None]
    T_flat = T_test.flatten()[:, None]
    
    U_pred = solver.predict(X_flat, T_flat)
    U_exact = solver.exact_solution(X_flat, T_flat)
    
    l2_error = np.linalg.norm(U_pred - U_exact) / np.linalg.norm(U_exact)
    max_error = np.max(np.abs(U_pred - U_exact))
    
    print(f"Relative L2 Error: {l2_error:.6f}")
    print(f"Maximum Absolute Error: {max_error:.6f}")
    print()
    
    # Visualize results
    print("Generating visualization...")
    fig = plot_solution(solver, save_path='outputs/heat_equation_solution.png')
    print()
    
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
