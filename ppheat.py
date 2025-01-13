import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HeatEquationNN(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=32):
        super().__init__()
        
        # Increased network capacity for more complex behavior
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ])
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)

def compute_pde_residual(model, x, t, alpha=0.01, cooling_rate=0.1):
    """
    Compute the residual of the heat equation with cooling to environment:
    du/dt = α(d²u/dx²) - k(u - u_ambient)
    where k is the cooling rate and u_ambient is ambient temperature (assumed to be 20°C)
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    u_ambient = 20.0  # Ambient temperature in Celsius
    
    du_dt = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    du_dx = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    d2u_dx2 = torch.autograd.grad(
        du_dx, x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]
    
    # Heat equation with cooling term
    return du_dt - alpha * d2u_dx2 + cooling_rate * (u - u_ambient)

def train_model(model, num_epochs=5000, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate training points
    x = torch.linspace(0, 1, 100).reshape(-1, 1)  # Rod length normalized to 1
    t = torch.linspace(0, 10, 100).reshape(-1, 1)  # Simulate for 10 time units
    X, T = torch.meshgrid(x.squeeze(), t.squeeze())
    x_train = X.reshape(-1, 1)
    t_train = T.reshape(-1, 1)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute PDE residual
        residual = compute_pde_residual(model, x_train, t_train)
        
        # Initial condition: temperature distribution at t=0
        # Left end hot (100°C), rest at ambient (20°C)
        x_ic = torch.linspace(0, 1, 100).reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        u_ic = model(x_ic, t_ic)
        initial_temp = 20 + 80 * torch.exp(-30 * x_ic)  # Exponential decay from hot end
        ic_loss = torch.mean((u_ic - initial_temp)**2)
        
        # Boundary conditions:
        # Both ends eventually cool to ambient temperature
        t_bc = torch.linspace(0, 10, 100).reshape(-1, 1)
        x_bc_left = torch.zeros_like(t_bc)
        x_bc_right = torch.ones_like(t_bc)
        
        # Natural cooling at boundaries
        u_bc_left = model(x_bc_left, t_bc)
        u_bc_right = model(x_bc_right, t_bc)
        bc_loss = torch.mean(
            torch.abs(u_bc_left - (20 + 80 * torch.exp(-0.5 * t_bc))) +  # Left end cooling
            torch.abs(u_bc_right - 20)  # Right end at ambient
        )
        
        # Total loss
        loss = torch.mean(residual**2) + ic_loss + bc_loss
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')
    
    return model

def plot_solution(model):
    """Plot both 2D and 3D visualizations of the solution with temperature color mapping"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(16, 6))
    
    # 2D plot at different times
    ax1 = fig.add_subplot(131)
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_steps = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Create custom colormap from red (hot) to blue (cold)
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(t_steps)))
    
    for t_val, color in zip(t_steps, colors):
        t = torch.full_like(x, t_val)
        u = model(x, t).detach().numpy()
        ax1.plot(x.numpy(), u, color=color, label=f't = {t_val}s', linewidth=2)
    
    ax1.set_xlabel('Position along rod (m)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Distribution Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # 3D surface plot
    ax2 = fig.add_subplot(132, projection='3d')
    
    x = torch.linspace(0, 1, 50)
    t = torch.linspace(0, 10, 50)
    X, T = torch.meshgrid(x, t)
    
    X_flat = X.flatten().reshape(-1, 1)
    T_flat = T.flatten().reshape(-1, 1)
    U = model(X_flat, T_flat).detach().reshape(X.shape)
    
    surf = ax2.plot_surface(X, T, U, cmap='RdYlBu_r', 
                          linewidth=0, antialiased=True)
    
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Time (s)')
    ax2.set_zlabel('Temperature (°C)')
    ax2.set_title('Heat Conduction Over Time')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Temperature (°C)')
    
    # Contour plot
    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(X, T, U, levels=20, cmap='RdYlBu_r')
    plt.colorbar(contour, label='Temperature (°C)')
    ax3.set_xlabel('Position (m)')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Temperature Contour Map')
    
    plt.tight_layout()
    plt.show()

# Create and train the model
model = HeatEquationNN()
trained_model = train_model(model)
plot_solution(trained_model)