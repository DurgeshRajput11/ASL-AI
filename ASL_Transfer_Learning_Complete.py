import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# Set up data directories
TRAIN_DIR = 'training'    # Path to your training data
VAL_DIR = 'validation'    # Path to your validation data (optional)

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 29  # A-Z + space + del + nothing
EPOCHS = 15

print("Starting ASL Transfer Learning with VGG16...")

# Step 1: Data Generators
print("Setting up data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
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

# Step 2: Create VGG16 Transfer Learning Model
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

# Step 3: Initial Training (with frozen base)
print("Starting initial training (base model frozen)...")
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

# Step 4: Fine-tuning
print("Setting up fine-tuning...")

# Unfreeze some layers for fine-tuning (e.g., last 5 VGG blocks)
fine_tune_at = 15  # Unfreeze from this layer onwards
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting fine-tuning...")
fine_tune_epochs = 5
total_epochs = EPOCHS + fine_tune_epochs

history_finetune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Step 5: Save Results
best_accuracy = max(history_finetune.history['val_accuracy'])
print("Training completed.")
print(f"Best Validation Accuracy: {best_accuracy:.2%}")
print("Model saved as: asl_vgg16_model.h5")

# Step 6: Save class names for later use
class_names = list(train_generator.class_indices.keys())
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)
print(f"Class names saved: {class_names}")

print("All done. Your ASL model is ready to use.")
