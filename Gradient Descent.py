import numpy as np
import matplotlib.pyplot as plt

# Normalizing the features
def feature_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

# Hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)

# Cost function with regularization
def cost(X, y, theta, lambda_reg):
    m = len(y)
    h = hypothesis(X, theta)
    cost = (1/(2*m)) * np.sum((h - y)**2) + (lambda_reg/(2*m)) * np.sum(theta[1:]**2)
    return cost

# Gradient function with regularization
def gradient(X, y, theta, lambda_reg):
    m = len(y)
    h = hypothesis(X, theta)
    grad = (1/m) * np.dot(X.T, (h - y))
    grad[1:] += (lambda_reg/m) * theta[1:]
    return grad

# Gradient Descent
def gradient_descent(X, y, alpha, num_iters, lambda_reg):
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    for iter in range(num_iters):
        theta = theta - alpha * gradient(X, y, theta, lambda_reg)
        J_history.append(cost(X, y, theta, lambda_reg))

        # Adaptive learning rate (optional)
        if iter > 0 and J_history[-2] - J_history[-1] < 1e-7:
            alpha *= 0.9

    return theta, J_history

# Example Usage
X = np.array([[2], [3], [4], [5]])
y = np.array([[5], [7], [9], [11]])
X, mean, std = feature_normalize(X)
X = np.hstack([np.ones((X.shape[0], 1)), X])

alpha = 0.01
num_iters = 1000
lambda_reg = 1  # Regularization parameter

theta, J_history = gradient_descent(X, y, alpha, num_iters, lambda_reg)
print("Optimized Theta:", theta)

# Plotting
plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.show()
