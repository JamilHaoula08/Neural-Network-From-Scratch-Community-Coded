import numpy as np
 
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
# Training data (inputs and expected outputs)
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[0], [1], [1], [0]])
 
# Random weights
weights = np.random.rand(2, 1)
 
# Train for 10,000 iterations
for _ in range(10000):
    prediction = sigmoid(np.dot(inputs, weights))
    error = outputs - prediction
    weights += np.dot(inputs.T, error * prediction * (1 - prediction))
 
print("Predictions after training:")
print(prediction)
 
