# Human Pose Detection App

A comprehensive human pose detection application that combines MoveNet inference with custom pose estimation training capabilities. This project provides both real-time pose detection using Google's MoveNet model and a custom-trained pose estimation model.

## 🚀 Features

- **Real-time Pose Detection**: Uses Google's MoveNet Lightning model for fast, accurate pose detection
- **Custom Model Training**: Train your own pose estimation model using COCO dataset
- **Multi-person Detection**: Supports detection of multiple people in a single frame
- **Visualization Tools**: Comprehensive visualization and testing utilities
- **GPU Support**: Optimized for GPU acceleration with TensorFlow
- **Data Augmentation**: Advanced data augmentation techniques for improved model training

## 📁 Project Structure

```
human-pose-detection-app/
├── movenet_infer.py          # MoveNet real-time inference
├── train_pose.py            # Custom pose model training
├── test_and_visualize.py    # Model testing and visualization
├── pose_model_best.h5       # Best trained model weights
├── pose_model_last.h5       # Last checkpoint weights
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

### Install Dependencies

```bash
pip install tensorflow tensorflow-hub opencv-python numpy matplotlib scikit-learn
```

### GPU Support (Optional)

For GPU acceleration, install TensorFlow with GPU support:

```bash
pip install tensorflow-gpu
```

## 🎯 Usage

### 1. Real-time Pose Detection with MoveNet

Run the MoveNet inference script for real-time pose detection:

```bash
python movenet_infer.py
```

**Features:**
- Randomly selects images from your dataset directory
- Displays pose keypoints and connections
- Press 'q' to quit
- Supports multiple people detection

**Configuration:**
Update the `IMAGES_DIR` path in `movenet_infer.py` to point to your image dataset.

### 2. Train Custom Pose Model

Train your own pose estimation model using COCO dataset:

```bash
python train_pose.py
```

**Features:**
- Uses MobileNetV2 as backbone
- Advanced data augmentation
- Early stopping and model checkpointing
- Configurable training parameters

**Configuration:**
Update these paths in `train_pose.py`:
- `IMAGES_DIR`: Path to your COCO images
- `ANNOTATIONS_FILE`: Path to COCO keypoints annotations

### 3. Test and Visualize Model

Test your trained model and visualize results:

```bash
python test_and_visualize.py
```

**Features:**
- Synthetic test image generation
- Multiple preprocessing method testing
- Real image testing
- Comprehensive visualization

## 🔧 Configuration

### MoveNet Configuration

The MoveNet model uses the following keypoints (17 points):
- 0: nose
- 1-2: left/right eye
- 3-4: left/right ear
- 5-6: left/right shoulder
- 7-8: left/right elbow
- 9-10: left/right wrist
- 11-12: left/right hip
- 13-14: left/right knee
- 15-16: left/right ankle

### Model Architecture

**Custom Pose Model:**
- Backbone: MobileNetV2 (pre-trained on ImageNet)
- Input size: 256x256 pixels
- Output: 34 keypoints (17 points × 2 coordinates)
- Data augmentation: rotation, scaling, flipping, brightness adjustment

## 📊 Performance

- **MoveNet**: Real-time inference (~30 FPS on GPU)
- **Custom Model**: Optimized for accuracy with data augmentation
- **Multi-person**: Supports up to 6 people simultaneously

## 🎨 Visualization

The application provides rich visualization capabilities:
- Keypoint detection with confidence scores
- Skeleton connections with color coding
- Real-time display with OpenCV
- Matplotlib-based analysis plots

## 🔍 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure TensorFlow-GPU is properly installed
2. **Memory Issues**: Reduce batch size in training script
3. **Path Issues**: Verify dataset paths are correct
4. **Model Loading**: Ensure model files are in the correct directory

### Performance Tips

- Use GPU for faster inference and training
- Adjust confidence threshold for different use cases
- Optimize image resolution based on your needs
- Use data augmentation for better model generalization

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please open an issue on the project repository.

---

**Note**: This project requires the COCO dataset for training. Please ensure you have proper access to the dataset and follow the COCO dataset license terms.
