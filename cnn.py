import tensorflow as tf
import numpy as np

# Define a custom activation function
def custom_activation(x):
    return tf.math.sin(tf.math.square(x))

# Define the CNN architecture
class CNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation=custom_activation, input_shape=(32, 32, 3))
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation=custom_activation)
        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation=custom_activation)
        self.conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation=custom_activation)
        
        # Max pooling layers
        self.pooling1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.pooling2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.pooling3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.pooling4 = tf.keras.layers.MaxPooling2D((2, 2))
        
        # Batch normalization layers
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        
        # Fully connected layers
        self.fc1 = tf.keras.layers.Dense(1024, activation=custom_activation)
        self.fc2 = tf.keras.layers.Dense(512, activation=custom_activation)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax')
        
        # Dropout layers for regularization
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dropout2 = tf.keras.layers.Dropout(0.4)

    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.pooling3(x)
        x = self.conv4(x)
        x = self.pooling4(x)
        x = tf.keras.layers.Flatten()(x)
        
        # Apply dropout for regularization
        x = self.dropout1(x)
        
        x = self.fc1(x)
        
        # Apply dropout for regularization
        x = self.dropout2(x)
        
        x = self.fc2(x)
        return self.fc3(x)

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

# Learning rate scheduling
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Data augmentation for training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Train the model with data augmentation
batch_size = 64
epochs = 40
model.fit(data_augmentation(X_train), y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
