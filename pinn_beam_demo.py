"""
Physics-Informed Neural Networks (PINNs) - Structural Analysis Demo
====================================================================
Solving the 1D Euler-Bernoulli Beam Deflection Equation

PDE: EI * d^4w/dx^4 = q(x)  (4th order ODE)

For a simply-supported beam with uniform load:
- Boundary Conditions: w(0) = w(L) = 0 (no deflection at supports)
- Boundary Conditions: w''(0) = w''(L) = 0 (no moment at supports)

Analytical Solution (uniform load q0, length L):
w(x) = (q0 * x) / (24 * E * I) * (L^3 - 2*L*x^2 + x^3)

This is a classic structural mechanics problem demonstrating:
- 4th order differential equations
- Multiple boundary condition types
- Engineering applications of PINNs

Author: Seymur Hasanov (PINN Exploration for Capstone)
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
L = 1.0  # Beam length (m)
E = 200e9  # Young's modulus (Pa) - Steel
I = 8.33e-6  # Moment of inertia (m^4) - for a 10cm x 10cm square section
q0 = 10000  # Uniform load (N/m)
EI = E * I  # Flexural rigidity

# Normalize for numerical stability
L_norm = L
EI_norm = 1.0  # We'll work with normalized equation
q_norm = q0 * L**4 / EI  # Normalized load

N_EPOCHS = 10000
LEARNING_RATE = 0.001
N_COLLOCATION = 1000
N_BOUNDARY = 100

print(f"Beam Properties:")
print(f"  Length: {L} m")
print(f"  E: {E/1e9:.1f} GPa")
print(f"  I: {I*1e6:.2f} mm^4 (×10^-6 m^4)")
print(f"  EI: {EI:.2f} N·m²")
print(f"  Load q0: {q0} N/m")
print(f"  Normalized load (q*L^4/EI): {q_norm:.4f}")

# ============================================================================
# Analytical Solution
# ============================================================================
def analytical_deflection(x, q=q0, length=L, flexural_rigidity=EI):
    """
    Analytical solution for simply-supported beam with uniform load.
    w(x) = (q * x) / (24 * EI) * (L^3 - 2*L*x^2 + x^3)
    """
    return (q * x / (24 * flexural_rigidity)) * (length**3 - 2*length*x**2 + x**3)

def analytical_deflection_normalized(x, q_n=q_norm):
    """Normalized analytical solution (L=1, EI=1)."""
    return (q_n * x / 24) * (1 - 2*x**2 + x**3)

def analytical_moment(x, q=q0, length=L):
    """Bending moment M(x) = EI * w''(x) = q/2 * (Lx - x^2)"""
    return (q / 2) * (length * x - x**2)

def analytical_shear(x, q=q0, length=L):
    """Shear force V(x) = EI * w'''(x) = q * (L/2 - x)"""
    return q * (length / 2 - x)

# ============================================================================
# PINN Implementation
# ============================================================================
class PINN_Beam(nn.Module):
    """Physics-Informed Neural Network for Beam Deflection."""
    
    def __init__(self, hidden_layers=[50, 50, 50, 50, 50]):
        super(PINN_Beam, self).__init__()
        layers = []
        input_dim = 1  # x only (static problem)
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def compute_beam_residual(model, x, q_normalized=q_norm):
    """
    Compute the beam equation residual: d^4w/dx^4 - q/EI = 0
    (Using normalized form: d^4w/dx^4 = q_normalized)
    """
    x.requires_grad_(True)
    
    w = model(x)
    
    # First derivative
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w),
                               create_graph=True, retain_graph=True)[0]
    
    # Second derivative
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x),
                                create_graph=True, retain_graph=True)[0]
    
    # Third derivative
    w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=torch.ones_like(w_xx),
                                 create_graph=True, retain_graph=True)[0]
    
    # Fourth derivative
    w_xxxx = torch.autograd.grad(w_xxx, x, grad_outputs=torch.ones_like(w_xxx),
                                  create_graph=True, retain_graph=True)[0]
    
    # Beam equation residual: d^4w/dx^4 = q (normalized)
    residual = w_xxxx - q_normalized
    
    return residual, w, w_xx

def train_pinn_beam():
    """Train the PINN model for beam deflection."""
    print("\n" + "=" * 60)
    print("Training PINN for Simply-Supported Beam Deflection")
    print("=" * 60)
    print(f"Governing Equation: EI * d⁴w/dx⁴ = q")
    print(f"Boundary Conditions:")
    print(f"  w(0) = 0, w(L) = 0  (no deflection)")
    print(f"  w''(0) = 0, w''(L) = 0  (no moment)")
    print("=" * 60 + "\n")
    
    # Initialize model
    model = PINN_Beam(hidden_layers=[50, 50, 50, 50, 50])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    # Collocation points (interior)
    x_coll = torch.rand(N_COLLOCATION, 1) * L_norm
    
    # Boundary points at x=0
    x_bc_0 = torch.zeros(N_BOUNDARY, 1)
    
    # Boundary points at x=L
    x_bc_L = torch.ones(N_BOUNDARY, 1) * L_norm
    
    # Training loop
    loss_history = []
    physics_history = []
    bc_history = []
    
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        
        # Physics loss
        residual, _, _ = compute_beam_residual(model, x_coll)
        loss_physics = torch.mean(residual**2)
        
        # Boundary conditions at x=0: w(0) = 0
        w_0 = model(x_bc_0)
        loss_w_0 = torch.mean(w_0**2)
        
        # Boundary conditions at x=L: w(L) = 0
        w_L = model(x_bc_L)
        loss_w_L = torch.mean(w_L**2)
        
        # Moment boundary conditions: w''(0) = 0, w''(L) = 0
        x_bc_0.requires_grad_(True)
        x_bc_L.requires_grad_(True)
        
        _, _, w_xx_0 = compute_beam_residual(model, x_bc_0)
        _, _, w_xx_L = compute_beam_residual(model, x_bc_L)
        
        loss_m_0 = torch.mean(w_xx_0**2)
        loss_m_L = torch.mean(w_xx_L**2)
        
        # Total loss
        loss_bc = loss_w_0 + loss_w_L + loss_m_0 + loss_m_L
        loss = loss_physics + 100 * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        physics_history.append(loss_physics.item())
        bc_history.append(loss_bc.item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
                  f"Physics: {loss_physics.item():.6f} | BC: {loss_bc.item():.6f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, (loss_history, physics_history, bc_history)

def evaluate_beam_model(model):
    """Evaluate model and compare with analytical solution."""
    x = np.linspace(0, L_norm, 200)
    
    # Analytical solution (normalized)
    w_analytical = analytical_deflection_normalized(x)
    
    # PINN prediction
    x_tensor = torch.tensor(x[:, None], dtype=torch.float32)
    with torch.no_grad():
        w_pinn = model(x_tensor).numpy().flatten()
    
    # Convert to physical units for display
    w_physical_analytical = w_analytical * EI / (q0 * L**3)  # Back to meters
    w_physical_pinn = w_pinn * EI / (q0 * L**3)
    
    # Error
    error = np.abs(w_analytical - w_pinn)
    
    return x, w_analytical, w_pinn, error

def create_beam_visualization(x, w_analytical, w_pinn, error, loss_histories):
    """Create comprehensive visualization of beam deflection results."""
    
    loss_history, physics_history, bc_history = loss_histories
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('PINN Solution: Euler-Bernoulli Beam Deflection\n' + 
                 r'$EI\frac{d^4w}{dx^4} = q(x)$ — Simply-Supported Beam with Uniform Load', 
                 fontsize=14, fontweight='bold')
    
    # 1. Deflection comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, w_analytical * 1000, 'b-', linewidth=2, label='Analytical')
    ax1.plot(x, w_pinn * 1000, 'r--', linewidth=2, label='PINN')
    ax1.set_xlabel('Position x (normalized)')
    ax1.set_ylabel('Deflection w (×10³)')
    ax1.set_title('Beam Deflection Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Deflection is typically shown downward
    
    # 2. Error plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, error * 1e6, 'g-', linewidth=2)
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Absolute Error (×10⁶)')
    ax2.set_title(f'Prediction Error (Max: {error.max():.2e})')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(x, 0, error * 1e6, alpha=0.3, color='green')
    
    # 3. Training loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(loss_history, 'b-', linewidth=1, alpha=0.7, label='Total Loss')
    ax3.semilogy(physics_history, 'r-', linewidth=1, alpha=0.7, label='Physics Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('Training Loss History')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Beam schematic
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(-0.1, 1.1)
    ax4.set_ylim(-0.3, 0.4)
    
    # Draw beam
    ax4.plot([0, 1], [0, 0], 'b-', linewidth=8, solid_capstyle='butt')
    
    # Draw supports (triangles)
    support_size = 0.08
    ax4.plot([0, -support_size/2, support_size/2, 0], 
             [0, -support_size, -support_size, 0], 'k-', linewidth=2)
    ax4.plot([1, 1-support_size/2, 1+support_size/2, 1], 
             [0, -support_size, -support_size, 0], 'k-', linewidth=2)
    
    # Draw distributed load (arrows)
    for xi in np.linspace(0.05, 0.95, 15):
        ax4.annotate('', xy=(xi, 0), xytext=(xi, 0.25),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax4.plot([0, 1], [0.25, 0.25], 'r-', linewidth=2)
    ax4.text(0.5, 0.32, f'q = {q0} N/m', ha='center', fontsize=10, color='red')
    
    # Labels
    ax4.text(-0.05, 0, 'A', fontsize=12, fontweight='bold', ha='right')
    ax4.text(1.05, 0, 'B', fontsize=12, fontweight='bold', ha='left')
    ax4.text(0.5, -0.25, f'L = {L} m', ha='center', fontsize=10)
    
    ax4.set_title('Problem Setup: Simply-Supported Beam')
    ax4.axis('equal')
    ax4.axis('off')
    
    # 5. Bending Moment Diagram
    ax5 = fig.add_subplot(gs[1, 1])
    M = analytical_moment(x * L)  # Physical moment
    ax5.fill_between(x, 0, M/1000, alpha=0.3, color='blue')
    ax5.plot(x, M/1000, 'b-', linewidth=2)
    ax5.set_xlabel('Position x')
    ax5.set_ylabel('Bending Moment M (kN·m)')
    ax5.set_title('Bending Moment Diagram')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 6. Shear Force Diagram
    ax6 = fig.add_subplot(gs[1, 2])
    V = analytical_shear(x * L)  # Physical shear
    ax6.fill_between(x, 0, V/1000, alpha=0.3, color='orange')
    ax6.plot(x, V/1000, 'orange', linewidth=2)
    ax6.set_xlabel('Position x')
    ax6.set_ylabel('Shear Force V (kN)')
    ax6.set_title('Shear Force Diagram')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 7. Physical deflection scale
    ax7 = fig.add_subplot(gs[2, 0])
    w_physical = analytical_deflection(x * L) * 1000  # in mm
    w_pinn_physical = w_pinn * EI / (q0 * L**3) * 1000  # in mm (approximate)
    
    # For display, scale PINN output
    scale_factor = w_physical.max() / (w_pinn.max() * 1000) if w_pinn.max() != 0 else 1
    
    ax7.plot(x, w_physical, 'b-', linewidth=2, label='Analytical (mm)')
    ax7.set_xlabel('Position x/L')
    ax7.set_ylabel('Deflection (mm)')
    ax7.set_title(f'Physical Deflection (max: {w_physical.max():.4f} mm)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.invert_yaxis()
    
    # 8. Convergence at different points
    ax8 = fig.add_subplot(gs[2, 1])
    points_to_check = [0.25, 0.5, 0.75]
    colors = ['red', 'green', 'blue']
    for i, xi in enumerate(points_to_check):
        idx = int(xi * len(x))
        ax8.scatter([xi], [w_pinn[idx] * 1000], color=colors[i], s=100, 
                    label=f'x={xi}: PINN', zorder=5, marker='o')
        ax8.scatter([xi], [w_analytical[idx] * 1000], color=colors[i], s=100,
                    label=f'x={xi}: Exact', zorder=5, marker='x')
    ax8.plot(x, w_analytical * 1000, 'k--', alpha=0.5, label='Full curve')
    ax8.set_xlabel('Position x')
    ax8.set_ylabel('Deflection (×10³)')
    ax8.set_title('Point-wise Comparison')
    ax8.legend(loc='upper left', fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.invert_yaxis()
    
    # 9. Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate metrics
    mse = np.mean(error**2)
    mae = np.mean(error)
    max_error = np.max(error)
    r2 = 1 - np.sum((w_analytical - w_pinn)**2) / np.sum((w_analytical - np.mean(w_analytical))**2)
    
    summary_text = f"""
    Beam Deflection PINN Results
    ============================
    
    Problem: Euler-Bernoulli Beam
    EI × d⁴w/dx⁴ = q(x)
    
    Beam Properties:
    - Length: L = {L} m
    - Young's Modulus: E = {E/1e9:.0f} GPa
    - Moment of Inertia: I = {I*1e6:.2f}×10⁻⁶ m⁴
    - Uniform Load: q = {q0} N/m
    
    Boundary Conditions:
    - Simply-supported (pin-pin)
    - w(0) = w(L) = 0
    - M(0) = M(L) = 0
    
    Accuracy Metrics:
    - R² Score: {r2:.6f}
    - Mean Squared Error: {mse:.2e}
    - Maximum Error: {max_error:.2e}
    
    Max Deflection (analytical):
    {analytical_deflection(L/2)*1000:.6f} mm at x=L/2
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    return fig, {'mse': mse, 'mae': mae, 'max_error': max_error, 'r2': r2}

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("PHYSICS-INFORMED NEURAL NETWORKS (PINNs)")
    print("Structural Analysis: Euler-Bernoulli Beam Deflection")
    print("="*60)
    
    # Train the model
    model, loss_histories = train_pinn_beam()
    
    # Evaluate
    x, w_analytical, w_pinn, error = evaluate_beam_model(model)
    
    # Visualize
    fig, metrics = create_beam_visualization(x, w_analytical, w_pinn, error, loss_histories)
    
    # Save results
    output_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/PINN_exploration/results/PINN_Beam_Results.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    pdf_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/PINN_exploration/results/PINN_Beam_Results.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("STRUCTURAL ANALYSIS CASE STUDY COMPLETE!")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  - R² Score: {metrics['r2']*100:.2f}%")
    print(f"  - Max Error: {metrics['max_error']:.2e}")
    print(f"\nThis demonstrates PINNs can solve 4th-order ODEs")
    print(f"from structural mechanics with high accuracy!")
    print("="*60)
