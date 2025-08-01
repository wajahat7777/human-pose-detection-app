import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load your model
model = tf.keras.models.load_model('pose_model_best.h5', compile=False)

print("=== POSE DISTINCTION TEST ===")

# Create synthetic test images with obvious different "poses"
# Image 1: Person-like shape in upper left
img1 = np.zeros((192, 192, 3), dtype=np.float32)
cv2.rectangle(img1, (30, 30), (70, 100), (1.0, 1.0, 1.0), -1)  # torso
cv2.circle(img1, (50, 20), 10, (1.0, 1.0, 1.0), -1)  # head
img1 = cv2.resize(img1, (256, 256))  # Resize to model input shape

# Image 2: Person-like shape in lower right  
img2 = np.zeros((192, 192, 3), dtype=np.float32)
cv2.rectangle(img2, (120, 120), (160, 180), (1.0, 1.0, 1.0), -1)  # torso
cv2.circle(img2, (140, 110), 10, (1.0, 1.0, 1.0), -1)  # head
img2 = cv2.resize(img2, (256, 256))  # Resize to model input shape

# Predict on both
pred1 = model.predict(np.expand_dims(img1, 0), verbose=0)[0].reshape(-1, 2)
pred2 = model.predict(np.expand_dims(img2, 0), verbose=0)[0].reshape(-1, 2)

# Check if predictions follow the input positions
pred1_center = pred1.mean(axis=0)
pred2_center = pred2.mean(axis=0)

print(f"Image 1 (upper-left shape) - predicted center: ({pred1_center[0]:.3f}, {pred1_center[1]:.3f})")
print(f"Image 2 (lower-right shape) - predicted center: ({pred2_center[0]:.3f}, {pred2_center[1]:.3f})")

# Convert to pixel coordinates for visualization
pred1_pixels = pred1 * 256
pred2_pixels = pred2 * 256

# Visualize the test
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Show synthetic images
axes[0,0].imshow(img1)
axes[0,0].set_title('Test Image 1 (Upper-Left)')
axes[0,0].axis('off')

axes[0,1].imshow(img2)
axes[0,1].set_title('Test Image 2 (Lower-Right)')
axes[0,1].axis('off')

# Show predictions
img1_with_pred = img1.copy()
for x, y in pred1_pixels:
    if 0 <= x < 256 and 0 <= y < 256:
        cv2.circle(img1_with_pred, (int(x), int(y)), 2, (0, 1, 0), -1)

img2_with_pred = img2.copy()
for x, y in pred2_pixels:
    if 0 <= x < 256 and 0 <= y < 256:
        cv2.circle(img2_with_pred, (int(x), int(y)), 2, (0, 1, 0), -1)

axes[1,0].imshow(img1_with_pred)
axes[1,0].set_title('Predictions on Image 1')
axes[1,0].axis('off')

axes[1,1].imshow(img2_with_pred)
axes[1,1].set_title('Predictions on Image 2')
axes[1,1].axis('off')

plt.tight_layout()
plt.show()

# Analysis
print(f"\nAnalysis:")
print(f"Center shift X: {pred2_center[0] - pred1_center[0]:.3f} (should be positive)")
print(f"Center shift Y: {pred2_center[1] - pred1_center[1]:.3f} (should be positive)")

if pred2_center[0] > pred1_center[0] and pred2_center[1] > pred1_center[1]:
    print("✅ GOOD: Model responds to spatial changes - might work with correct preprocessing")
    
    print("\n=== TRYING DIFFERENT PREPROCESSING ON REAL IMAGE ===")
    
    # Test on a real image with different preprocessing
    sample_image_path = r'C:\Users\MR LAPTOP\Desktop\val2017\val2017\000000000139.jpg'
    
    try:
        img = cv2.imread(sample_image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            # Try multiple preprocessing approaches
            preprocessing_methods = {
                'Method 1 - Simple /255': lambda x: cv2.resize(x, (192, 192)) / 255.0,
                'Method 2 - ImageNet': lambda x: (cv2.resize(x, (192, 192)) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225],
                'Method 3 - Zero Center': lambda x: (cv2.resize(x, (192, 192)) / 255.0) - 0.5,
                'Method 4 - Std Norm': lambda x: (cv2.resize(x, (192, 192)) - 127.5) / 127.5,
            }
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # Original image
            axes[0].imshow(img_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            best_method = None
            best_spread = 0
            
            for i, (method_name, preprocess_fn) in enumerate(preprocessing_methods.items()):
                try:
                    # Preprocess
                    processed = preprocess_fn(img_rgb)
                    input_batch = np.expand_dims(processed, axis=0)
                    
                    # Predict
                    pred = model.predict(input_batch, verbose=0)[0]
                    pred_kps = pred.reshape(-1, 2) * [w, h]  # Scale to original image size
                    
                    # Calculate spread of predictions (measure of diversity)
                    spread = np.std(pred_kps, axis=0).mean()
                    
                    if spread > best_spread:
                        best_spread = spread
                        best_method = method_name
                    
                    # Visualize
                    img_with_pred = img_rgb.copy()
                    for x, y in pred_kps:
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(img_with_pred, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    axes[i+1].imshow(img_with_pred)
                    axes[i+1].set_title(f'{method_name}\nSpread: {spread:.1f}')
                    axes[i+1].axis('off')
                    
                    print(f"{method_name}: Prediction spread = {spread:.1f}")
                    
                except Exception as e:
                    print(f"Error with {method_name}: {e}")
                    axes[i+1].text(0.5, 0.5, f'Error\n{method_name}', ha='center', va='center')
                    axes[i+1].axis('off')
            
            # Hide unused subplot
            if len(axes) > len(preprocessing_methods) + 1:
                axes[-1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n🎯 BEST METHOD: {best_method} (highest spread = {best_spread:.1f})")
            print("Higher spread means keypoints are more distributed across the image")
            
        else:
            print("Could not load sample image")
            
    except Exception as e:
        print(f"Error testing real image: {e}")
        
else:
    print("❌ BAD: Model doesn't respond to spatial changes")
    print("Your model likely wasn't trained properly and needs retraining")
    
    print("\n🔧 RETRAINING RECOMMENDATIONS:")
    print("1. Use MSE loss function (not categorical crossentropy)")
    print("2. Train for at least 50-100 epochs")
    print("3. Use proper data augmentation")
    print("4. Monitor training/validation loss curves")
    print("5. Expect final training loss < 0.01 for good results")

# === VISUALIZING ON REAL DATASET IMAGES ===
import os
import json

IMAGES_DIR = r'C:\Users\MR LAPTOP\Desktop\val2017\val2017'
ANNOTATIONS_FILE = r'C:\Users\MR LAPTOP\Desktop\annotations_trainval2017\annotations\person_keypoints_val2017.json'

with open(ANNOTATIONS_FILE, 'r') as f:
    coco = json.load(f)

id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

shown = 0
for ann in coco['annotations']:
    if ann['num_keypoints'] < 5:
        continue
    img_path = os.path.join(IMAGES_DIR, id_to_filename[ann['image_id']])
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (256, 256)) / 255.0
    input_img = np.expand_dims(img_resized, 0)
    pred = model.predict(input_img, verbose=0)[0].reshape(-1, 2)
    kps = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
    kps = kps / [orig_w, orig_h]
    pred_pixels = pred * [orig_w, orig_h]
    gt_pixels = kps * [orig_w, orig_h]
    vis_img = img_rgb.copy()
    for x, y in pred_pixels:
        cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)
    for x, y in gt_pixels:
        cv2.circle(vis_img, (int(x), int(y)), 3, (255, 0, 0), 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_img)
    plt.title(f'Green: Predicted, Red: Ground Truth\n{os.path.basename(img_path)}')
    plt.axis('off')
    plt.show()
    shown += 1
    if shown >= 5:
        break