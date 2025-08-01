import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== USER: UPDATE THESE PATHS ====
IMAGES_DIR = r'C:\Users\MR LAPTOP\Desktop\val2017\val2017'  # e.g., './val2017'
ANNOTATIONS_FILE = r'C:\Users\MR LAPTOP\Desktop\annotations_trainval2017\annotations\person_keypoints_val2017.json'  # e.g., './annotations/person_keypoints_val2017.json'

# Load COCO keypoints annotations
with open(ANNOTATIONS_FILE, 'r') as f:
    coco = json.load(f)

# Build image id to file name mapping
id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

# Prepare data: images and keypoints
images = []
keypoints = []

for i, ann in enumerate(coco['annotations']):
    if ann['num_keypoints'] < 5:  # skip images with too few keypoints
        continue
    img_path = os.path.join(IMAGES_DIR, id_to_filename[ann['image_id']])
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Increased input size for better resolution
    images.append(img / 255.0)  # Normalize

    # Keypoints: [x1, y1, v1, x2, y2, v2, ...] (v=visibility)
    kps = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]  # Only x, y
    # Normalize x, y to [0,1] based on original image size
    orig_img = cv2.imread(img_path)
    h, w = orig_img.shape[:2]
    kps = kps / [w, h]
    kps = kps.flatten()
    keypoints.append(kps)
    if (i+1) % 500 == 0:
        print(f"Loaded {i+1} annotations...")

if len(images) == 0:
    raise RuntimeError("No images loaded! Check your dataset paths and annotation file.")

images = np.array(images, dtype=np.float32)
keypoints = np.array(keypoints, dtype=np.float32)

print(f"Loaded {len(images)} samples.")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, keypoints, test_size=0.1, random_state=42, shuffle=True
)

# Data augmentation (advanced) for training only
train_datagen = ImageDataGenerator(
    rotation_range=20,         # rotate images by up to 20 degrees
    width_shift_range=0.1,     # shift horizontally by up to 10%
    height_shift_range=0.1,    # shift vertically by up to 10%
    zoom_range=0.1,            # zoom in/out by up to 10%
    brightness_range=[0.8,1.2],# random brightness
    horizontal_flip=True,      # randomly flip images
    shear_range=10,            # shear transformation
    channel_shift_range=20,    # random channel shifts
    fill_mode='nearest'        # fill mode for transforms
)
val_datagen = ImageDataGenerator()  # No augmentation for validation

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=16,
    shuffle=True
)
val_generator = val_datagen.flow(
    X_val, y_val,
    batch_size=16,
    shuffle=False
)

# Improved model: MobileNetV2 backbone with enhanced head
base_model = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3),
                                               include_top=False,
                                               weights='imagenet',
                                               pooling='avg')
x = layers.Dense(256, activation='relu')(base_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(34, activation='sigmoid')(x)  # 17 keypoints * 2 (x, y)
model = models.Model(inputs=base_model.input, outputs=output)

# Print model summary
model.summary()

# Compile with MSE loss for keypoint regression
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Callbacks for best training practices
checkpoint = ModelCheckpoint(
    'pose_model_best.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)
callbacks = [checkpoint, early_stop, reduce_lr]

# Train with generators and callbacks
history = model.fit(
    train_generator,
    epochs=200,
    validation_data=val_generator,
    callbacks=callbacks
)

# Save final model (last epoch)
model.save('pose_model_last.h5')
print("Model saved as pose_model_last.h5 (last epoch) and pose_model_best.h5 (best val loss)")

# Plot training and validation loss curves
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()