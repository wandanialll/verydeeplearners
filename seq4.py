import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Generate synthetic data for linear regression
np.random.seed(42)
X = np.random.rand(10000, 1)  # Increased dataset size
y = 3 * X + 2 + np.random.randn(10000, 1) * 0.1

# Define the deep neural network model
model = Sequential()
model.add(Dense(50, input_dim=1, activation='relu'))  # Increased number of neurons
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=200, batch_size=32)  # Increased number of epochs and batch size

# Evaluate the model
loss = model.evaluate(X, y)
print(f'Loss: {loss}')

# Make predictions
predictions = model.predict(X)
print(f'Predictions: {predictions[:5]}')

# Plot the results
plt.scatter(X, y, label='Data Points', alpha=0.3)
plt.plot(X, predictions, color='red', label='Model Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data Points and Model Predictions')
plt.legend()
plt.show()
# Get the weights of the model
weights = model.get_weights()

# The first layer's weights and bias
coefficients = weights[0]
intercept = weights[1]

print(f'Coefficient: {coefficients[0][0]}')
print(f'Intercept: {intercept[0]}')