import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define your dataset paths
# NOTE: Replace 'path/to/your/dataset/train' and 'path/to/your/dataset/test'
# with the actual paths to your training and testing image directories.
train_dir = 'path/to/your/dataset/train'
test_dir = 'path/to/your/dataset/test'

# --- 1. Data Preprocessing and Augmentation ---

# Image parameters
image_size = (128, 128) # Standardized image size
batch_size = 32
input_shape = image_size + (3,) # Assuming RGB images (3 channels). Use (1,) for grayscale.

# Training Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

# Test Data Preprocessing (only rescaling, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
# 'class_mode' should be 'binary' for 2 classes (e.g., tumor/normal)
# or 'categorical' for 3 or more classes.
class_mode_type = 'binary' 

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode_type
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode_type
)

# Determine the number of classes and output activation
if class_mode_type == 'binary':
    num_classes = 1
    output_activation = 'sigmoid' # For binary classification
    loss_function = 'binary_crossentropy'
else:
    num_classes = len(training_set.class_indices)
    output_activation = 'softmax' # For multi-class classification
    loss_function = 'categorical_crossentropy'


# --- 2. Build the CNN Model ---

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flattening
    Flatten(),
    
    # Fully Connected Layers
    Dense(units=512, activation='relu'),
    Dropout(0.5), # Add dropout to prevent overfitting
    Dense(units=num_classes, activation=output_activation)
])

# --- 3. Compile the Model ---

model.compile(
    optimizer='adam',
    loss=loss_function,
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()


# --- 4. Train the Model ---

# Use 'steps_per_epoch' and 'validation_steps' to define a full pass
# (number of batches) over the dataset for training and validation.
history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // batch_size,
    epochs=25, # You
