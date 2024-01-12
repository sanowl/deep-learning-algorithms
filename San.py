import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# Define the Spatial Transformer Network Layer
class SpatialTransformerNetwork(tf.keras.layers.Layer):
    # [Existing STN code with enhancements...]
    pass

# Define the Advanced Model
def create_advanced_model(input_shape, num_classes):
    model = models.Sequential([
        SpatialTransformerNetwork(output_size=(32, 32)),  # Enhanced STN Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        # [Additional layers...]
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess dataset
# [Dataset loading code...]
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(X_val, y_val, batch_size=32)

# Compile and Train the Model
model = create_advanced_model(input_shape, num_classes)
model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(train_generator, epochs=150, validation_data=validation_generator, callbacks=[lr_scheduler, early_stopping])

# Evaluate the Model
from sklearn.metrics import classification_report

predictions = model.predict(test_datagen.flow(X_test, batch_size=32))
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(y_test, predicted_classes))
