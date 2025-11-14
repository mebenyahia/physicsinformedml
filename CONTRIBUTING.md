# Contributing to Physics-Informed Machine Learning

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant code snippets or error messages

### Suggesting Enhancements

We welcome suggestions for new features or improvements! Please open an issue with:
- A clear description of the enhancement
- Why this would be useful
- Examples of how it would work
- Any relevant references or papers

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines below
3. **Add tests** if you're adding functionality
4. **Update documentation** if needed
5. **Ensure tests pass** by running the examples
6. **Submit a pull request**

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions and classes (Google style)
- Keep functions focused and modular
- Use meaningful variable names

### Documentation
- Update README.md if adding new features
- Add examples to demonstrate new functionality
- Include mathematical explanations for new PDEs
- Use clear, concise language

### Example Code Structure

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
    """
    # Implementation
    return result
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/mebenyahia/physicsinformedml.git
cd physicsinformedml
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run tests:
```bash
python examples/quick_demo.py
```

## Adding New PDE Examples

When adding a new PDE solver:

1. **Create a new file** in `src/` (e.g., `pinn_wave_equation.py`)
2. **Implement the solver** following the structure of `pinn_heat_equation.py`
3. **Add documentation** explaining:
   - The PDE being solved
   - Boundary and initial conditions
   - Any exact solutions (if available)
   - Physical interpretation
4. **Create an example** in `examples/`
5. **Update README.md** with the new example

### Template for New PDE Solvers

```python
class NewEquationPINN:
    """
    PINN solver for [Name of PDE].
    
    Solves: [Mathematical equation]
    """
    
    def __init__(self, ...):
        """Initialize solver with parameters."""
        pass
    
    def pde_loss(self, x, t):
        """Calculate PDE residual loss."""
        pass
    
    def boundary_loss(self, ...):
        """Calculate boundary condition loss."""
        pass
    
    def train(self, ...):
        """Train the PINN."""
        pass
    
    def predict(self, x, t):
        """Make predictions."""
        pass
```

## Areas for Contribution

We particularly welcome contributions in these areas:

### New PDE Examples
- Wave equation
- Burgers' equation
- Navier-Stokes equations
- Schrödinger equation
- Diffusion-reaction equations
- Maxwell's equations

### Advanced Features
- Adaptive collocation point sampling
- Transfer learning between similar problems
- Uncertainty quantification
- Multi-fidelity approaches
- GPU optimization
- Parallel training strategies

### Documentation
- More tutorial notebooks
- Video tutorials
- Blog posts about the theory
- Translation to other languages
- Better visualization tools

### Testing
- Unit tests for core functionality
- Integration tests
- Performance benchmarks
- Numerical accuracy tests

## Code Review Process

1. All submissions require review
2. We aim to review PRs within 1 week
3. Feedback will be provided for improvements
4. Once approved, changes will be merged

## Questions?

Feel free to open an issue with the label "question" if you need help or clarification.

## Recognition

Contributors will be recognized in:
- README.md acknowledgments section
- Git commit history
- Release notes

Thank you for contributing! 🎉
