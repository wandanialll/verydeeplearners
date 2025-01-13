import numpy as np
from scipy.integrate import odeint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

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

# Define the neural network model with increased layers
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
model.compile(optimizer='adam', loss='mse')

# Start the timer
start_time = time.time()

# Train the model with increased epochs and batch size
model.fit(X_train, y_train, epochs=2000, batch_size=40, verbose=1)  # Increased epochs

# Stop the timer
end_time = time.time()

# Calculate the duration
duration = end_time - start_time
print(f"Training time: {duration:.2f} seconds")

# Predict using the trained model
f_pred = model.predict(X_train)

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
