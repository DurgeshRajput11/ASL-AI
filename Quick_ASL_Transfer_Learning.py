import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# Set up data directories
TRAIN_DIR = 'training'    
VAL_DIR = 'validation'   

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 29  
EPOCHS = 15

print("Starting ASL Transfer Learning with VGG16...")

print("Setting up data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"Found {train_generator.samples} training images")
print(f"Found {val_generator.samples} validation images")
print(f"Classes: {train_generator.num_classes}")
print("Creating VGG16 transfer learning model...")

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created successfully.")
print(f"Total parameters: {model.count_params():,}")
print("Starting training...")
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'asl_vgg16_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

#  Save Results
best_accuracy = max(history.history['val_accuracy'])
print("Training completed.")
print(f"Best Validation Accuracy: {best_accuracy:.2%}")
print("Model saved as: asl_vgg16_model.h5")

class_names = list(train_generator.class_indices.keys())
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)
print(f"Class names saved: {class_names}")

print("All done. Your ASL model is ready to use.")
