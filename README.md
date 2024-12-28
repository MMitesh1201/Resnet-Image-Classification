
# Real-Time Object Classification with ResNet50V2

A robust real-time object classification system using the ResNet50V2 architecture, fine-tuned for improved accuracy and performance.

## Authors

- [@MMitesh1201](https://github.com/MMitesh1201)


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mitesh-agrawal-9934ab215/)

## Features

- Real-time object classification using webcam feed
- Pre-trained ResNet50V2 architecture with custom fine-tuning
- Support for multiple object classes
- Performance optimization for real-time processing
- Comprehensive evaluation metrics and visualization
- Easy-to-use inference pipeline

## Tech Stack

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Pandas
- CUDA (for GPU acceleration)
- Docker

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/realtime-object-classification.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Model Architecture

The system uses ResNet50V2 as the backbone architecture, which features:
- Skip connections to mitigate vanishing gradient problems
- Batch normalization for improved training stability
- Pre-activation structure for better information flow
- 50 layers deep with over 23 million trainable parameters

## Fine-Tuning Process

### Phase 1: Feature Extraction

- Freeze all ResNet50V2 layers except the final classification layers
- Train new classification head on custom dataset
- Learning rate: 1e-4 with Adam optimizer
- Batch size: 32
- Data augmentation:
  - Random horizontal flip
  - Random rotation (±15 degrees)
  - Random brightness adjustment
  - Random zoom (0.9-1.1)

### Phase 2: Deep Fine-Tuning

- Unfreeze final ResNet50V2 blocks
- Lower learning rate to 1e-5
- Implement gradual unfreezing
- Use learning rate scheduling
- Apply weight decay for regularization
- Implement early stopping with patience=5

## Usage

### Training

```bash
# Start training with default parameters
python train.py

# Custom training configuration
python train.py --batch_size 64 --epochs 50 --learning_rate 0.0001
```


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- ResNet50V2 paper authors for the original architecture
- TensorFlow team for the framework and pre-trained models
- The open-source community for various contributions
