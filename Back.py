import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0) * 1

# Mean Squared Error Loss Function
def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Layer class
class Layer:
    def __init__(self, size, activation_function, activation_derivative):
        self.weights = np.random.randn(size[0], size[1]) * np.sqrt(2. / size[0])
        self.bias = np.random.randn(size[1], 1)
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.output = None
        self.input = None
        self.error = None
        self.delta = None

    def activate(self, x):
        self.input = x
        self.output = self.activation_function(np.dot(x, self.weights) + self.bias.T)
        return self.output

# Neural Network class
class NeuralNetwork:
    def __init__(self, learning_rate=0.1, lambda_reg=0.01, momentum=0.9):
        self.layers = []
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.momentum = momentum
        self.velocity = []

    def add_layer(self, layer):
        self.layers.append(layer)
        self.velocity.append(np.zeros_like(layer.weights))

    def feed_forward(self, X):
        for layer in self.layers:
            X = layer.activate(X)
        return X

    def back_propagation(self, X, y):
        # Forward pass
        self.feed_forward(X)

        # Backward pass
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                # Output layer
                layer.error = y - layer.output
                layer.delta = layer.error * layer.activation_derivative(layer.output)
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
                layer.delta = layer.error * layer.activation_derivative(layer.output)

            # Update weights and biases with momentum
            grad = np.dot(layer.input.T, layer.delta)
            reg_adjustment = self.lambda_reg * layer.weights
            self.velocity[i] = self.momentum * self.velocity[i] + grad
            layer.weights += self.learning_rate * self.velocity[i] - reg_adjustment
            layer.bias += self.learning_rate * np.sum(layer.delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, batch_size=32, validation_data=None, early_stopping_rounds=None):
        best_loss = np.inf
        rounds_without_improvement = 0

        for epoch in range(epochs):
            # Mini-batch training
            for X_batch, y_batch in self.get_batches(X, y, batch_size):
                self.back_propagation(X_batch, y_batch)

            # Learning rate decay
            if epoch % 100 == 0 and epoch != 0:
                self.learning_rate /= 2

            # Training loss
            train_loss = mse_loss(y, self.feed_forward(X))
            message = f"Epoch: {epoch}, Training Loss: {train_loss}"

            # Validation evaluation
            if validation_data is not None:
                val_loss = mse_loss(validation_data[1], self.feed_forward(validation_data[0]))
                message += f", Validation Loss: {val_loss}"

                # Early stopping
                if early_stopping_rounds is not None:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        rounds_without_improvement = 0
                    else:
                        rounds_without_improvement += 1

                    if rounds_without_improvement == early_stopping_rounds:
                        print("Early stopping triggered")
                        break

            print(message)

    def get_batches(self, X, y, batch_size):
        for i in range(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size], y[i:i + batch_size])

# Example usage
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(learning_rate=0.1)
nn.add_layer(Layer((2, 3), relu, relu_derivative))  # Hidden layer 1
nn.add_layer(Layer((3, 1), sigmoid, sigmoid_derivative))  # Output layer

validation_data = (X, y)  # Typically, you would use separate validation data
nn.train(X, y, epochs=2000, batch_size=2, validation_data=validation_data, early_stopping_rounds=10)
