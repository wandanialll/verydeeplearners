"""
Group: VERY DEEP LEARNERS

Members:
Wan Muhammad Danial Bin Zulkifli (81475)
Muhammad Faaris Bin Jamhari (78196)

Project Title: Distributed Learning of Deep Neural Networks

*** CODE EXPLANATION ***
            #############################
THIS IS THE ##SEQUENTIAL IMPLEMENTATION## OF THE VERY DEEP LEARNERS PROJECT.
            #############################

Input: 
- No user input required.
- Parameters such as the number of epochs, batch size, and learning rate can be adjusted within the code.
- Differential equations for the prey-predator model are defined in the function df1.
- Initial values for the ODE are arbitrary and can be changed within the code.

Output: 
- The code outputs a plot of the prey-predator model using the ODE solver and the neural network predictions.
- It also prints the training time for the neural network and the total execution time.

Compilation and execution instructions:
1. Ensure numpy, scipy, tensorflow, and matplotlib are installed.
2. Run the code using the following command:
    python VERYDEEPLEARNERS_Sequential.py
    **Note: Check Python version and package installation paths if issues arise.
"""

import numpy as np
from scipy.integrate import odeint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

# Start the total timer
overall_start_time = time.time()

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

# Neural network design with multiple layers
model = Sequential()
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))  
model.add(Dense(128, activation='relu'))  
model.add(Dense(128, activation='relu'))  
model.add(Dense(2))

# Model compilation with Adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mse')

# Start the training timer
start_time = time.time()

# Train the model with increased epochs and batch size
model.fit(X_train, y_train, epochs=2000, batch_size=40, verbose=1)  # Increased epochs

# Stop the training timer
end_time = time.time()

# Calculate the training duration
training_duration = end_time - start_time
print(f"Training time: {training_duration:.2f} seconds")

# Predict using the trained model
f_pred = model.predict(X_train)

# Stop the overall timer
overall_end_time = time.time()

# Calculate the total duration
total_duration = overall_end_time - overall_start_time
print(f"Total execution time: {total_duration:.2f} seconds")

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