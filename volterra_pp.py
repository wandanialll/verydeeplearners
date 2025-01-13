from mpi4py import MPI
import numpy as np
from scipy.integrate import odeint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start the timer for the entire script
start_time = time.time()

# Define the differential equations
def df1(f, t):
    b = 1.0
    a = 0.5
    c = 0.01
    d = 0.005
    dfdt = [b * f[0] - c * f[0] * f[1], d * f[0] * f[1] - a * f[1]]
    return dfdt

# Split the time range for parallel computation
f0 = [200, 80]
t_total = 20
t_points = 101
t_full = np.linspace(0, t_total, t_points)
split_size = t_points // size

if rank == size - 1:
    t_chunk = t_full[rank * split_size:]
else:
    t_chunk = t_full[rank * split_size:(rank + 1) * split_size + 1]

# Start the timer for ODE solving
ode_start_time = time.time()

# Each process solves a part of the ODE
f_chunk = odeint(df1, f0, t_chunk)

# End the timer for ODE solving
ode_end_time = time.time()
ode_elapsed_time = ode_end_time - ode_start_time
print(f"Rank {rank}: ODE solving time: {ode_elapsed_time:.2f} seconds")

# Gather the ODE results at the root process
f_all = comm.gather(f_chunk, root=0)
t_all = comm.gather(t_chunk, root=0)

if rank == 0:
    # Combine the ODE results
    t_combined = np.concatenate(t_all)
    f_combined = np.vstack(f_all)

    # Prepare training data
    X_train = t_combined.reshape(-1, 1)
    y_train = f_combined

    # Define the neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=1, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))  # Added layer
    model.add(Dense(128, activation='relu'))  # Added layer
    model.add(Dense(2))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Start the timer for neural network training
    nn_start_time = time.time()

    # Train the model
    model.fit(X_train, y_train, epochs=2000, batch_size=40, verbose=0)  # Reduce epochs for testing

    # End the timer for neural network training
    nn_end_time = time.time()
    nn_elapsed_time = nn_end_time - nn_start_time
    print(f"Neural network training time: {nn_elapsed_time:.2f} seconds")

    # Predict using the trained model
    f_pred = model.predict(X_train, verbose=0)

    # End the timer for the entire script
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Plot the results
    plt.plot(t_combined, f_combined[:, 0], 'r-.>', label='Prey (ODE)')
    plt.plot(t_combined, f_combined[:, 1], '.b-', label='Predator (ODE)')
    plt.plot(t_combined, f_pred[:, 0], 'g--', label='Prey (NN)')
    plt.plot(t_combined, f_pred[:, 1], 'm--', label='Predator (NN)')
    plt.xlabel('t (years)')
    plt.ylabel('Population density')
    plt.title('Prey vs Predator population')
    plt.legend(loc='upper left')
    plt.show()