from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train_parallel_model():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    epochs = 5

    # Transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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

    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Rank {rank}: Starting training...")
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Gather the model weights from all processes
        for param in model.parameters():
            comm.Allreduce(MPI.IN_PLACE, param.data, op=MPI.SUM)
            param.data /= size

        print(f"Rank {rank}: Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

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
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_parallel_model()
