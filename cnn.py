import tensorflow as tf
import numpy as np

# Define a custom activation function
def custom_activation(x):
    return tf.math.sin(tf.math.square(x))

# Define the CNN architecture
class AdvancedCNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(AdvancedCNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation=custom_activation, padding='same', input_shape=(32, 32, 3))
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation=custom_activation, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation=custom_activation, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation=custom_activation, padding='same')
        
        # Batch normalization layers
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        
        # Spatial Dropout layers
        self.spatial_dropout1 = tf.keras.layers.SpatialDropout2D(0.3)
        self.spatial_dropout2 = tf.keras.layers.SpatialDropout2D(0.3)
        self.spatial_dropout3 = tf.keras.layers.SpatialDropout2D(0.3)
        self.spatial_dropout4 = tf.keras.layers.SpatialDropout2D(0.3)
        
        # Global Average Pooling
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        
        # Fully connected layers with skip connections
        self.fc1 = tf.keras.layers.Dense(1024, activation=custom_activation)
        self.fc2 = tf.keras.layers.Dense(512, activation=custom_activation)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax')
        
        # Dropout layers for regularization
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dropout2 = tf.keras.layers.Dropout(0.4)

    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.spatial_dropout1(x)
        
        x_skip1 = x  # Skip connection 1
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.spatial_dropout2(x)
        
        x_skip2 = x  # Skip connection 2
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.spatial_dropout3(x)
        
        x_skip3 = x  # Skip connection 3
        
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.spatial_dropout4(x)
        
        x_skip4 = x  # Skip connection 4
        
        x = self.global_average_pooling(x)
        
        # Apply dropout for regularization
        x = self.dropout1(x)
        
        x = self.fc1(x)
        
        # Apply dropout for regularization
        x = self.dropout2(x)
        
        x = self.fc2(x)
        
        # Skip connection addition
        x = x + x_skip1 + x_skip2 + x_skip3 + x_skip4
        
        return self.fc3(x)

# Load and preprocess your dataset (e.g., CIFAR-10)
# Replace this with your dataset loading and preprocessing code
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values to [0, 1]

# Define the number of classes in your dataset
num_classes = 10  # Example: 10 classes for CIFAR-10

# Create an instance of the advanced model
model = AdvancedCNNModel(num_classes)

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

# Train the advanced model with data augmentation
batch_size = 64
epochs = 60
model.fit(data_augmentation(X_train), y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the advanced model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
