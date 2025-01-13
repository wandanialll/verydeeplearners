import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate training data: random pairs of (x, y)
X_train = np.random.uniform(-2, 2, size=(1000, 2))  # Random points in a range
x_train = X_train[:, 0]
y_train = X_train[:, 1]

# Define the system of equations
def system_of_equations(x, y):
    eq1 = x**2 + y**2 - 1  # x^2 + y^2 = 1
    eq2 = x + y - 1        # x + y = 1
    return eq1, eq2

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2)  # Output will be two values, one for each equation
])

# Custom loss function: Sum of squares of both equations
def custom_loss(y_true, y_pred):
    eq1_loss = tf.reduce_mean(tf.square(y_pred[:, 0]))  # Error for first equation
    eq2_loss = tf.reduce_mean(tf.square(y_pred[:, 1]))  # Error for second equation
    return eq1_loss + eq2_loss

# Compile the model
model.compile(optimizer=Adam(), loss=custom_loss)

# Train the model
model.fit(X_train, np.zeros((X_train.shape[0], 2)), epochs=500, batch_size=64)

# Test the model with new inputs
test_input = np.array([[0.5, 0.5]])  # A starting guess
output = model.predict(test_input)

# Output predictions
print("Predicted x, y values for input [0.5, 0.5]:", output)
print("Predicted values of the system of equations (should be close to zero):", output[0])

