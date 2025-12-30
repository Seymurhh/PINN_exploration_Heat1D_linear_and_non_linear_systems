"""
Physics-Informed Neural Networks (PINNs) MVP Demo
==================================================
Solving the 1D Heat Equation using PINNs

PDE: du/dt = alpha * d^2u/dx^2
Domain: x in [0, 1], t in [0, 1]
Boundary Conditions: u(0, t) = u(1, t) = 0
Initial Condition: u(x, 0) = sin(pi * x)

Analytical Solution: u(x, t) = sin(pi * x) * exp(-alpha * pi^2 * t)

Author: Seymur Hasanov (MVP for Capstone Exploration)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using numpy-based simplified demo.")

# ============================================================================
# Configuration
# ============================================================================
ALPHA = 0.1  # Thermal diffusivity
N_EPOCHS = 5000
LEARNING_RATE = 0.001
N_COLLOCATION = 10000  # Points to enforce physics
N_BOUNDARY = 200  # Boundary condition points
N_INITIAL = 200  # Initial condition points

# ============================================================================
# Analytical Solution (Ground Truth)
# ============================================================================
def analytical_solution(x, t, alpha=ALPHA):
    """Exact solution to the 1D heat equation with given BC/IC."""
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

# ============================================================================
# PINN Implementation (PyTorch)
# ============================================================================
if TORCH_AVAILABLE:
    
    class PINN(nn.Module):
        """Physics-Informed Neural Network for 1D Heat Equation."""
        
        def __init__(self, hidden_layers=[50, 50, 50, 50]):
            super(PINN, self).__init__()
            layers = []
            input_dim = 2  # (x, t)
            
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.Tanh())
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, 1))  # Output: u(x, t)
            self.network = nn.Sequential(*layers)
        
        def forward(self, x, t):
            inputs = torch.cat([x, t], dim=1)
            return self.network(inputs)
    
    def compute_pde_residual(model, x, t, alpha=ALPHA):
        """Compute the PDE residual: du/dt - alpha * d^2u/dx^2 = 0"""
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = model(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        
        # PDE residual: du/dt - alpha * d^2u/dx^2
        residual = u_t - alpha * u_xx
        return residual
    
    def train_pinn():
        """Train the PINN model."""
        print("=" * 60)
        print("Training Physics-Informed Neural Network")
        print("=" * 60)
        
        # Initialize model
        model = PINN(hidden_layers=[50, 50, 50, 50])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Generate training points
        # Collocation points (interior domain)
        x_coll = torch.rand(N_COLLOCATION, 1)
        t_coll = torch.rand(N_COLLOCATION, 1)
        
        # Boundary conditions: u(0, t) = u(1, t) = 0
        x_bc_left = torch.zeros(N_BOUNDARY, 1)
        x_bc_right = torch.ones(N_BOUNDARY, 1)
        t_bc = torch.rand(N_BOUNDARY, 1)
        u_bc = torch.zeros(N_BOUNDARY, 1)
        
        # Initial condition: u(x, 0) = sin(pi * x)
        x_ic = torch.rand(N_INITIAL, 1)
        t_ic = torch.zeros(N_INITIAL, 1)
        u_ic = torch.sin(np.pi * x_ic)
        
        # Training loop
        loss_history = []
        
        for epoch in range(N_EPOCHS):
            optimizer.zero_grad()
            
            # Physics loss (PDE residual)
            residual = compute_pde_residual(model, x_coll, t_coll)
            loss_physics = torch.mean(residual**2)
            
            # Boundary condition loss
            u_pred_left = model(x_bc_left, t_bc)
            u_pred_right = model(x_bc_right, t_bc)
            loss_bc = torch.mean(u_pred_left**2) + torch.mean(u_pred_right**2)
            
            # Initial condition loss
            u_pred_ic = model(x_ic, t_ic)
            loss_ic = torch.mean((u_pred_ic - u_ic)**2)
            
            # Total loss
            loss = loss_physics + 10 * loss_bc + 10 * loss_ic
            
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
                      f"Physics: {loss_physics.item():.6f} | "
                      f"BC: {loss_bc.item():.6f} | IC: {loss_ic.item():.6f}")
        
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return model, loss_history
    
    def evaluate_model(model):
        """Evaluate model and compare with analytical solution."""
        # Create evaluation grid
        x = np.linspace(0, 1, 100)
        t = np.linspace(0, 1, 100)
        X, T = np.meshgrid(x, t)
        
        # Analytical solution
        U_analytical = analytical_solution(X, T)
        
        # PINN prediction
        x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
        t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32)
        
        with torch.no_grad():
            u_pred = model(x_flat, t_flat).numpy()
        
        U_pinn = u_pred.reshape(X.shape)
        
        # Error
        error = np.abs(U_analytical - U_pinn)
        
        return X, T, U_analytical, U_pinn, error

# ============================================================================
# Visualization
# ============================================================================
def create_visualization(X, T, U_analytical, U_pinn, error, loss_history):
    """Create comprehensive visualization of results."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Physics-Informed Neural Networks (PINNs) - 1D Heat Equation MVP', 
                 fontsize=16, fontweight='bold')
    
    # 1. Analytical Solution
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(X, T, U_analytical, levels=50, cmap='hot')
    ax1.set_xlabel('x (Position)')
    ax1.set_ylabel('t (Time)')
    ax1.set_title('Analytical Solution')
    plt.colorbar(c1, ax=ax1, label='u(x,t)')
    
    # 2. PINN Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.contourf(X, T, U_pinn, levels=50, cmap='hot')
    ax2.set_xlabel('x (Position)')
    ax2.set_ylabel('t (Time)')
    ax2.set_title('PINN Prediction')
    plt.colorbar(c2, ax=ax2, label='u(x,t)')
    
    # 3. Absolute Error
    ax3 = fig.add_subplot(gs[0, 2])
    c3 = ax3.contourf(X, T, error, levels=50, cmap='Blues')
    ax3.set_xlabel('x (Position)')
    ax3.set_ylabel('t (Time)')
    ax3.set_title(f'Absolute Error (Max: {error.max():.4f})')
    plt.colorbar(c3, ax=ax3, label='|Error|')
    
    # 4. Training Loss
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(loss_history)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Training Loss History')
    ax4.grid(True, alpha=0.3)
    
    # 5. Slice at t=0 (Initial Condition)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(X[0, :], U_analytical[0, :], 'b-', linewidth=2, label='Analytical')
    ax5.plot(X[0, :], U_pinn[0, :], 'r--', linewidth=2, label='PINN')
    ax5.set_xlabel('x')
    ax5.set_ylabel('u(x, t=0)')
    ax5.set_title('Initial Condition (t = 0)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Slice at t=0.5
    t_idx = 50  # t = 0.5
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(X[t_idx, :], U_analytical[t_idx, :], 'b-', linewidth=2, label='Analytical')
    ax6.plot(X[t_idx, :], U_pinn[t_idx, :], 'r--', linewidth=2, label='PINN')
    ax6.set_xlabel('x')
    ax6.set_ylabel('u(x, t=0.5)')
    ax6.set_title('Solution at t = 0.5')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Slice at x=0.5
    x_idx = 50  # x = 0.5
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(T[:, x_idx], U_analytical[:, x_idx], 'b-', linewidth=2, label='Analytical')
    ax7.plot(T[:, x_idx], U_pinn[:, x_idx], 'r--', linewidth=2, label='PINN')
    ax7.set_xlabel('t')
    ax7.set_ylabel('u(x=0.5, t)')
    ax7.set_title('Temperature at x = 0.5 over Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 3D Surface Plot - PINN
    ax8 = fig.add_subplot(gs[2, 1], projection='3d')
    ax8.plot_surface(X, T, U_pinn, cmap='hot', alpha=0.8)
    ax8.set_xlabel('x')
    ax8.set_ylabel('t')
    ax8.set_zlabel('u')
    ax8.set_title('PINN Solution (3D)')
    
    # 9. Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate metrics
    mse = np.mean(error**2)
    mae = np.mean(error)
    max_error = np.max(error)
    r2 = 1 - np.sum((U_analytical - U_pinn)**2) / np.sum((U_analytical - np.mean(U_analytical))**2)
    
    summary_text = f"""
    PINN MVP Results Summary
    ========================
    
    Problem: 1D Heat Equation
    du/dt = alpha * d²u/dx²
    
    Domain: x ∈ [0, 1], t ∈ [0, 1]
    Alpha (thermal diffusivity): {ALPHA}
    
    Training:
    - Epochs: {N_EPOCHS}
    - Collocation Points: {N_COLLOCATION}
    - Learning Rate: {LEARNING_RATE}
    
    Accuracy Metrics:
    - Mean Squared Error: {mse:.6f}
    - Mean Absolute Error: {mae:.6f}
    - Maximum Error: {max_error:.6f}
    - R² Score: {r2:.6f}
    
    Conclusion:
    The PINN successfully learned to solve
    the heat equation from physics alone,
    with {r2*100:.2f}% accuracy!
    """
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig, {'mse': mse, 'mae': mae, 'max_error': max_error, 'r2': r2}

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    
    if TORCH_AVAILABLE:
        print("\n" + "="*60)
        print("PHYSICS-INFORMED NEURAL NETWORKS (PINNs) - MVP DEMO")
        print("="*60)
        print("\nProblem: 1D Heat Equation")
        print("du/dt = alpha * d²u/dx²")
        print(f"alpha = {ALPHA}")
        print("\nThis demonstrates how neural networks can learn to solve")
        print("differential equations by embedding physics into the loss function.")
        print("="*60 + "\n")
        
        # Train the model
        model, loss_history = train_pinn()
        
        # Evaluate
        X, T, U_analytical, U_pinn, error = evaluate_model(model)
        
        # Visualize
        fig, metrics = create_visualization(X, T, U_analytical, U_pinn, error, loss_history)
        
        # Save results
        output_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/PINN_MVP_Results.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nResults saved to: {output_path}")
        
        # Also save as PDF
        pdf_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/PINN_MVP_Results.pdf"
        fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")
        
        plt.show()
        
        print("\n" + "="*60)
        print("MVP COMPLETE!")
        print("="*60)
        print(f"\nKey Results:")
        print(f"  - R² Score: {metrics['r2']*100:.2f}%")
        print(f"  - Max Error: {metrics['max_error']:.4f}")
        print(f"\nThis proves PINNs can learn physics from the equation itself!")
        print("="*60)
        
    else:
        print("Please install PyTorch to run this demo:")
        print("  pip install torch")
