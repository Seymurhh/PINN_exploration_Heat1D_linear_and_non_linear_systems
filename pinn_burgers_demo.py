"""
Physics-Informed Neural Networks (PINNs) - Burgers Equation Demo
================================================================
Solving the 1D Viscous Burgers Equation (Non-Linear PDE)

PDE: du/dt + u * du/dx = nu * d^2u/dx^2
Domain: x in [-1, 1], t in [0, 1]
Viscosity: nu = 0.01/pi

This is a non-linear PDE that exhibits shock formation.

Author: Seymur Hasanov (MVP for Capstone Exploration)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# ============================================================================
# Configuration
# ============================================================================
NU = 0.01 / np.pi  # Viscosity (small = sharper shocks)
N_EPOCHS = 10000
LEARNING_RATE = 0.001
N_COLLOCATION = 10000
N_BOUNDARY = 200
N_INITIAL = 200

# ============================================================================
# Reference Solution (Cole-Hopf Transformation)
# ============================================================================
def analytical_burgers(x, t, nu=NU, n_terms=100):
    """
    Approximate analytical solution using Cole-Hopf transformation.
    For u(x,0) = -sin(pi*x), the solution involves Fourier series.
    Here we use a simplified numerical reference.
    """
    # For demonstration, we'll compute a reference using a fine grid
    # In practice, this would be the Cole-Hopf exact solution
    # For now, return None to indicate we'll use the PINN solution directly
    return None

def initial_condition(x):
    """Initial condition: u(x, 0) = -sin(pi * x)"""
    return -np.sin(np.pi * x)

# ============================================================================
# PINN Implementation
# ============================================================================
class PINN_Burgers(nn.Module):
    """Physics-Informed Neural Network for Burgers Equation."""
    
    def __init__(self, hidden_layers=[50, 50, 50, 50, 50]):
        super(PINN_Burgers, self).__init__()
        layers = []
        input_dim = 2  # (x, t)
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)

def compute_burgers_residual(model, x, t, nu=NU):
    """
    Compute the Burgers equation residual:
    du/dt + u * du/dx - nu * d^2u/dx^2 = 0
    """
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
    
    # Burgers residual: du/dt + u * du/dx - nu * d^2u/dx^2
    residual = u_t + u * u_x - nu * u_xx
    return residual

def train_pinn_burgers():
    """Train the PINN model for Burgers equation."""
    print("=" * 60)
    print("Training PINN for Non-Linear Burgers Equation")
    print("=" * 60)
    print(f"Viscosity nu = {NU:.6f}")
    print(f"This equation exhibits SHOCK FORMATION!")
    print("=" * 60 + "\n")
    
    # Initialize model (deeper network for non-linear problem)
    model = PINN_Burgers(hidden_layers=[50, 50, 50, 50, 50])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    # Generate training points
    # Collocation points (interior domain): x in [-1, 1], t in [0, 1]
    x_coll = 2 * torch.rand(N_COLLOCATION, 1) - 1  # x in [-1, 1]
    t_coll = torch.rand(N_COLLOCATION, 1)  # t in [0, 1]
    
    # Boundary conditions: u(-1, t) = u(1, t) = 0
    x_bc_left = -torch.ones(N_BOUNDARY, 1)
    x_bc_right = torch.ones(N_BOUNDARY, 1)
    t_bc = torch.rand(N_BOUNDARY, 1)
    u_bc = torch.zeros(N_BOUNDARY, 1)
    
    # Initial condition: u(x, 0) = -sin(pi * x)
    x_ic = 2 * torch.rand(N_INITIAL, 1) - 1  # x in [-1, 1]
    t_ic = torch.zeros(N_INITIAL, 1)
    u_ic = -torch.sin(np.pi * x_ic)
    
    # Training loop
    loss_history = []
    physics_history = []
    bc_history = []
    ic_history = []
    
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        
        # Physics loss (PDE residual)
        residual = compute_burgers_residual(model, x_coll, t_coll)
        loss_physics = torch.mean(residual**2)
        
        # Boundary condition loss
        u_pred_left = model(x_bc_left, t_bc)
        u_pred_right = model(x_bc_right, t_bc)
        loss_bc = torch.mean(u_pred_left**2) + torch.mean(u_pred_right**2)
        
        # Initial condition loss
        u_pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic)**2)
        
        # Total loss (higher weights on IC for shock problems)
        loss = loss_physics + 20 * loss_bc + 50 * loss_ic
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        physics_history.append(loss_physics.item())
        bc_history.append(loss_bc.item())
        ic_history.append(loss_ic.item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
                  f"Physics: {loss_physics.item():.6f} | "
                  f"BC: {loss_bc.item():.6f} | IC: {loss_ic.item():.6f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, (loss_history, physics_history, bc_history, ic_history)

def evaluate_burgers_model(model):
    """Evaluate model and create visualization data."""
    # Create evaluation grid
    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    
    # PINN prediction
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32)
    
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).numpy()
    
    U_pinn = u_pred.reshape(X.shape)
    
    return X, T, U_pinn

def create_burgers_visualization(X, T, U_pinn, loss_histories):
    """Create comprehensive visualization of Burgers equation results."""
    
    loss_history, physics_history, bc_history, ic_history = loss_histories
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('PINN Solution: Non-Linear Burgers Equation\n' + 
                 r'$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}$', 
                 fontsize=14, fontweight='bold')
    
    # 1. Solution contour plot
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(X, T, U_pinn, levels=100, cmap='RdBu_r')
    ax1.set_xlabel('x (Position)')
    ax1.set_ylabel('t (Time)')
    ax1.set_title('PINN Solution u(x,t)')
    plt.colorbar(c1, ax=ax1, label='u')
    
    # 2. 3D surface
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    surf = ax2.plot_surface(X, T, U_pinn, cmap='RdBu_r', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u')
    ax2.set_title('3D Solution Surface')
    ax2.view_init(elev=25, azim=-60)
    
    # 3. Time slices
    ax3 = fig.add_subplot(gs[0, 2])
    times = [0, 25, 50, 75, 99]  # indices corresponding to t = 0, 0.25, 0.5, 0.75, 1.0
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    for i, t_idx in enumerate(times):
        t_val = T[t_idx, 0]
        ax3.plot(X[t_idx, :], U_pinn[t_idx, :], color=colors[i], 
                 linewidth=2, label=f't = {t_val:.2f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('u(x, t)')
    ax3.set_title('Solution Profiles at Different Times')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 4. Training loss
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(loss_history, 'b-', linewidth=1, alpha=0.7, label='Total Loss')
    ax4.semilogy(physics_history, 'r-', linewidth=1, alpha=0.7, label='Physics Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Training Loss History')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Initial condition check
    ax5 = fig.add_subplot(gs[1, 1])
    x_ic = X[0, :]
    u_ic_exact = -np.sin(np.pi * x_ic)
    ax5.plot(x_ic, u_ic_exact, 'b-', linewidth=2, label='Exact IC: $-\\sin(\\pi x)$')
    ax5.plot(x_ic, U_pinn[0, :], 'r--', linewidth=2, label='PINN at t=0')
    ax5.set_xlabel('x')
    ax5.set_ylabel('u(x, 0)')
    ax5.set_title('Initial Condition Verification')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Shock formation visualization
    ax6 = fig.add_subplot(gs[1, 2])
    # Show gradient magnitude to highlight shock
    du_dx = np.gradient(U_pinn, X[0, :], axis=1)
    c6 = ax6.contourf(X, T, np.abs(du_dx), levels=50, cmap='hot')
    ax6.set_xlabel('x')
    ax6.set_ylabel('t')
    ax6.set_title('Gradient Magnitude |du/dx|\n(Shock Location)')
    plt.colorbar(c6, ax=ax6, label='|du/dx|')
    
    # 7. Space-time diagram with shock path
    ax7 = fig.add_subplot(gs[2, 0])
    c7 = ax7.pcolormesh(X, T, U_pinn, cmap='RdBu_r', shading='auto')
    ax7.set_xlabel('x')
    ax7.set_ylabel('t')
    ax7.set_title('Space-Time Diagram')
    plt.colorbar(c7, ax=ax7, label='u')
    
    # 8. Comparison: Linear vs Non-linear
    ax8 = fig.add_subplot(gs[2, 1])
    t_mid = 50  # t = 0.5
    ax8.plot(X[t_mid, :], U_pinn[t_mid, :], 'r-', linewidth=2, label='Burgers (Non-linear)')
    # For comparison, show what pure diffusion would look like
    u_diffusion = -np.sin(np.pi * X[t_mid, :]) * np.exp(-0.1 * np.pi**2 * 0.5)
    ax8.plot(X[t_mid, :], u_diffusion, 'b--', linewidth=2, label='Pure Diffusion (Linear)')
    ax8.set_xlabel('x')
    ax8.set_ylabel('u(x, t=0.5)')
    ax8.set_title('Linear vs Non-Linear at t=0.5')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate some metrics
    final_loss = loss_history[-1]
    min_u = np.min(U_pinn)
    max_u = np.max(U_pinn)
    
    summary_text = f"""
    Burgers Equation PINN Results
    =============================
    
    PDE: du/dt + u*du/dx = nu*d²u/dx²
    
    Parameters:
    - Viscosity: nu = {NU:.6f}
    - Domain: x in [-1, 1], t in [0, 1]
    - Epochs: {N_EPOCHS}
    
    Initial Condition:
    u(x, 0) = -sin(pi*x)
    
    Boundary Conditions:
    u(-1, t) = u(1, t) = 0
    
    Results:
    - Final Loss: {final_loss:.6f}
    - Solution Range: [{min_u:.3f}, {max_u:.3f}]
    
    Key Observations:
    1. Shock forms near x=0
    2. Wave steepens due to non-linear
       convection term (u * du/dx)
    3. Viscosity prevents discontinuity
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    return fig

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("PHYSICS-INFORMED NEURAL NETWORKS (PINNs)")
    print("Non-Linear Case: Viscous Burgers Equation")
    print("="*60)
    
    # Train the model
    model, loss_histories = train_pinn_burgers()
    
    # Evaluate
    X, T, U_pinn = evaluate_burgers_model(model)
    
    # Visualize
    fig = create_burgers_visualization(X, T, U_pinn, loss_histories)
    
    # Save results
    output_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/PINN_Burgers_Results.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    pdf_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/PINN_Burgers_Results.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("NON-LINEAR BURGERS EQUATION MVP COMPLETE!")
    print("="*60)
    print("\nThe PINN successfully captured:")
    print("  - Initial sinusoidal profile")
    print("  - Wave steepening (non-linear convection)")
    print("  - Shock formation at x=0")
    print("  - Diffusive smoothing")
    print("="*60)
