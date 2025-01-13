from mpi4py import MPI
import numpy as np
from scipy.integrate import odeint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Define the differential equations
def df1(f, t):
    b = 1.0
    a = 0.5
    c = 0.01
    d = 0.005
    dfdt = [b * f[0] - c * f[0] * f[1], d * f[0] * f[1] - a * f[1]]
    return dfdt

# Generate training data using the ODE solver
f0 = [200, 80]
tspan = np.linspace(0, 20, 101)
f = odeint(df1, f0, tspan)

# Prepare the training data
X_train = tspan.reshape(-1, 1)
y_train = f

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Split the data across different MPI processes
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

# Create the model and compile it
model = create_model()
optimizer = Adam(learning_rate=0.001)  # Try a lower learning rate

# Training parameters
epochs = 2000
batch_size = 40

# Gradient Clipping (to prevent large gradients)
clip_value = 1.0  # Clip gradients to this value

# Start the timer
start_time = time.time()

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(X_train_local, training=True)
        loss = tf.reduce_mean(tf.square(predictions - y_train_local))
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Clip gradients
    gradients = [tf.clip_by_value(grad, -clip_value, clip_value) for grad in gradients]

    # Use MPI to average gradients across all processes
    avg_gradients = [np.zeros_like(g.numpy()) for g in gradients]
    for i, gradient in enumerate(gradients):
        comm.Allreduce(gradient.numpy(), avg_gradients[i], op=MPI.SUM)
        avg_gradients[i] /= size  # Average the gradients
    
    # Apply the averaged gradients
    for i, variable in enumerate(model.trainable_variables):
        variable.assign_sub(optimizer.learning_rate * avg_gradients[i])
    
    # Print progress
    if rank == 0 and epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.numpy():.4f}")

# Stop the timer
end_time = time.time()
duration = end_time - start_time

# Only the root process prints the duration
if rank == 0:
    print(f"Training time: {duration:.2f} seconds")

# Gather predictions from all processes
predictions_all = comm.gather(model(X_train_local, training=False).numpy(), root=0)

# Only the root process will plot the results
if rank == 0:
    # Concatenate all predictions from different processes
    f_pred = np.concatenate(predictions_all, axis=0)
    
    # Ensure that the length of f_pred matches the length of tspan (101 points)
    if f_pred.shape[0] != len(tspan):
        raise ValueError(f"Prediction array length ({f_pred.shape[0]}) does not match tspan length ({len(tspan)})")
    
    # Plot the results
    plt.plot(tspan, f[:, 0], 'r-.>', label='Prey (ODE)')
    plt.plot(tspan, f[:, 1], '.b-', label='Predator (ODE)')
    plt.plot(tspan, f_pred[:, 0], 'g--', label='Prey (NN)')
    plt.plot(tspan, f_pred[:, 1], 'm--', label='Predator (NN)')
    plt.xlabel('t (years)')
    plt.ylabel('Population density')
    plt.title('Prey vs Predator population')
    plt.legend(loc='upper left')
    plt.show()
