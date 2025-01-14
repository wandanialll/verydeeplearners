from mpi4py import MPI
import numpy as np
from scipy.integrate import odeint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Start the overall timer to measure the total execution time
overall_start_time = time.time()

# Define the differential equations for the prey-predator model
def df1(f, t):
    b = 1.0
    a = 0.5
    c = 0.01
    d = 0.005
    dfdt = [b * f[0] - c * f[0] * f[1], d * f[0] * f[1] - a * f[1]]
    return dfdt

# Generate training data by solving the ODEs
f0 = [200, 80]
tspan = np.linspace(0, 20, 101)
f = odeint(df1, f0, tspan)

# Prepare the training data for the neural network
X_train = tspan.reshape(-1, 1)
y_train = f

# Initialize MPI for parallel processing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Split the training data among MPI processes
num_data_per_process = len(X_train) // size
start_index = rank * num_data_per_process
end_index = (rank + 1) * num_data_per_process if rank != size - 1 else len(X_train)

X_train_local = X_train[start_index:end_index]
y_train_local = y_train[start_index:end_index]

# Define the neural network model with multiple layers
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=1, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))  # Added layer
    model.add(Dense(128, activation='relu'))  # Added layer
    model.add(Dense(2))  # Output layer for 2 variables
    return model

# Create and compile the neural network model
model = create_model()
optimizer = Adam(learning_rate=0.001)  # Using a lower learning rate

# Set training parameters
epochs = 2000
batch_size = 40

# Set gradient clipping value to prevent large gradients
clip_value = 1.0

# Start the training timer to measure training duration
start_time = time.time()

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Forward pass: compute predictions
        predictions = model(X_train_local, training=True)
        loss = tf.reduce_mean(tf.square(predictions - y_train_local))
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Clip gradients to prevent large updates
    gradients = [tf.clip_by_value(grad, -clip_value, clip_value) for grad in gradients]

    # Use MPI to average gradients across all processes
    avg_gradients = [np.zeros_like(g.numpy()) for g in gradients]
    for i, gradient in enumerate(gradients):
        comm.Allreduce(gradient.numpy(), avg_gradients[i], op=MPI.SUM)
        avg_gradients[i] /= size  # Average the gradients
    
    # Apply the averaged gradients to update model weights
    for i, variable in enumerate(model.trainable_variables):
        variable.assign_sub(optimizer.learning_rate * avg_gradients[i])
    
    # Print progress every 100 epochs on the root process
    if rank == 0 and epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.numpy():.4f}")

# Stop the training timer
end_time = time.time()
training_duration = end_time - start_time

# Only the root process prints the training duration
if rank == 0:
    print(f"Training time: {training_duration:.2f} seconds")

# Gather predictions from all MPI processes
predictions_all = comm.gather(model(X_train_local, training=False).numpy(), root=0)

# Only the root process will handle results and plot
if rank == 0:
    # Concatenate all predictions from different processes
    f_pred = np.concatenate(predictions_all, axis=0)
    
    # Ensure the length of f_pred matches the length of tspan
    if f_pred.shape[0] != len(tspan):
        raise ValueError(f"Prediction array length ({f_pred.shape[0]}) does not match tspan length ({len(tspan)})")
    
    # Stop the overall timer
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print(f"Total execution time: {total_duration:.2f} seconds")
    
    # Plot the results comparing ODE and neural network predictions
    plt.plot(tspan, f[:, 0], 'r-.>', label='Prey (ODE)')
    plt.plot(tspan, f[:, 1], '.b-', label='Predator (ODE)')
    plt.plot(tspan, f_pred[:, 0], 'g--', label='Prey (NN)')
    plt.plot(tspan, f_pred[:, 1], 'm--', label='Predator (NN)')
    plt.xlabel('t (years)')
    plt.ylabel('Population density')
    plt.title('Prey vs Predator population')
    plt.legend(loc='upper left')
    plt.show()
