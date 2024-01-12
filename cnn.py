import tensorflow as tf
import numpy as np

# Define the CNN architecture
class CNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        
        # Max pooling layers
        self.pooling1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.pooling2 = tf.keras.layers.MaxPooling2D((2, 2))
        
        # Fully connected layers
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc1(x)
        return self.fc2(x)

# Load and preprocess your dataset (e.g., CIFAR-10)
# Replace this with your dataset loading and preprocessing code
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values to [0, 1]

# Define the number of classes in your dataset
num_classes = 10  # Example: 10 classes for CIFAR-10

# Create an instance of the model
model = CNNModel(num_classes)

# Define loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
