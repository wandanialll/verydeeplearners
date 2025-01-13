from mpi4py import MPI
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime

class HeatEquationNN(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=32):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)

def compute_pde_residual(model, x, t, alpha=0.01, cooling_rate=0.1):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    u_ambient = 20.0
    
    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]
    
    return du_dt - alpha * d2u_dx2 + cooling_rate * (u - u_ambient)

class MetricsTracker:
    def __init__(self, rank):
        self.rank = rank
        self.metrics = {
            'epoch': [],
            'total_loss': [],
            'computation_time': [],
            'communication_time': []
        }
    
    def update(self, epoch, loss, comp_time, comm_time):
        self.metrics['epoch'].append(epoch)
        self.metrics['total_loss'].append(loss)
        self.metrics['computation_time'].append(comp_time)
        self.metrics['communication_time'].append(comm_time)
    
    def save_to_excel(self):
        if self.rank == 0:  # Only rank 0 saves metrics
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'mpi_training_metrics_{timestamp}.xlsx'
            df = pd.DataFrame(self.metrics)
            df.to_excel(filename, index=False)
            print(f"Metrics saved to {filename}")

def train_mpi(num_epochs=5000):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create model and optimizer
    model = HeatEquationNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = MetricsTracker(rank)
    
    # Generate full dataset
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    t = torch.linspace(0, 10, 100).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_full = X.reshape(-1, 1)
    t_full = T.reshape(-1, 1)
    
    # Split data among processes
    local_size = len(x_full) // size
    start_idx = rank * local_size
    end_idx = start_idx + local_size if rank != size-1 else len(x_full)
    
    local_x = x_full[start_idx:end_idx]
    local_t = t_full[start_idx:end_idx]
    
    if rank == 0:
        print(f"Training started with {size} processes")
        print(f"Data points per process: {local_size}")
    
    for epoch in range(num_epochs):
        
        # Computation phase
        optimizer.zero_grad()
        comp_start = time.time()
        
        # Forward pass and loss computation
        residual = compute_pde_residual(model, local_x, local_t)
        local_loss = torch.mean(residual**2)
        
        # Initial condition (only include in rank 0's computation)
        if rank == 0:
            x_ic = torch.linspace(0, 1, 100).reshape(-1, 1)
            t_ic = torch.zeros_like(x_ic)
            u_ic = model(x_ic, t_ic)
            initial_temp = 20 + 80 * torch.exp(-30 * x_ic)
            ic_loss = torch.mean((u_ic - initial_temp)**2)
            local_loss += ic_loss
        
        # Backward pass
        local_loss.backward()
        
        comp_time = time.time() - comp_start
        comm_start = time.time()
        
        # All-reduce the gradients
        for param in model.parameters():
            grad_tensor = param.grad.data.numpy()
            grad_tensor = param.grad.data.numpy().astype(np.float64)
            global_grad = np.zeros_like(grad_tensor, dtype=np.float64)
            comm.Allreduce([grad_tensor, MPI.DOUBLE],
                          [global_grad, MPI.DOUBLE],
                          op=MPI.SUM)
            param.grad.data = torch.from_numpy(global_grad / size).float()
        # All-reduce the loss
        global_loss = comm.allreduce(local_loss.item(), op=MPI.SUM) / size
        
        comm_time = time.time() - comm_start
        
        # Update parameters
        optimizer.step()
        
        # Track metrics
        metrics.update(epoch, global_loss, comp_time, comm_time)
        
        # Print progress (only from rank 0)
        if rank == 0 and (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Loss: {global_loss:.6f}')
            print(f'  Computation time: {comp_time:.3f}s')
            print(f'  Communication time: {comm_time:.3f}s')
    
    # Save metrics at the end of training
    metrics.save_to_excel()
    
    # Gather the trained model at rank 0
    if rank == 0:
        return model
    return None

def plot_solution(model):
    """Plot solution if we're on rank 0"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(16, 6))
    
    # 2D plot
    ax1 = fig.add_subplot(131)
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_steps = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
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

if __name__ == "__main__":
    # This will run when executed with: mpirun -n <num_processes> python script.py
    model = train_mpi()
    
    # Only rank 0 plots the solution
    if MPI.COMM_WORLD.Get_rank() == 0:
        plot_solution(model)