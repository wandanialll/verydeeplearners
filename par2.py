from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def sync_parameters(model, comm):
    """Safely synchronize model parameters across processes."""
    try:
        for param in model.parameters():
            # Move parameter to CPU and convert to numpy array
            param_numpy = param.data.cpu().numpy()
            
            # Create a buffer of the same size
            buffer = np.zeros_like(param_numpy)
            
            # Gather parameters from all processes
            comm.Allreduce(param_numpy, buffer, op=MPI.SUM)
            
            # Average the parameters
            buffer = buffer / comm.Get_size()
            
            # Update the model parameter
            param.data = torch.from_numpy(buffer).to(param.device)
    except Exception as e:
        print(f"Error in parameter synchronization: {str(e)}")
        comm.Abort()

def train_parallel_model():
    try:
        # MPI setup
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Enhanced hyperparameters
        batch_size = 64  # Reduced batch size
        learning_rate = 0.001
        epochs = 10  # Reduced epochs
        
        # Basic transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the dataset
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        
        # Split the dataset based on rank
        data_per_rank = len(dataset) // size
        start_idx = rank * data_per_rank
        end_idx = (rank + 1) * data_per_rank if rank != size - 1 else len(dataset)
        subset_indices = list(range(start_idx, end_idx))
        subset = Subset(dataset, subset_indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # Initialize model, loss function, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        print(f"Rank {rank}: Starting training on {device}...")
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1

                # Synchronize less frequently (every 10 batches)
                if batch_count % 10 == 0:
                    sync_parameters(model, comm)

            avg_loss = epoch_loss / len(train_loader)
            print(f"Rank {rank}: Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Synchronize at the end of each epoch
            sync_parameters(model, comm)

        end_time = time.time()
        if rank == 0:
            print(f"Training completed in {end_time - start_time:.2f} seconds.")

        # Test the model on the root process
        if rank == 0:
            test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f"Test Accuracy: {100 * correct / total:.2f}%")

    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        comm.Abort()

if __name__ == "__main__":
    train_parallel_model()