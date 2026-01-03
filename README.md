# CIFAR-10 Classification: CNN vs. Vision Transformer (ViT) 🧠🖼️

This repository features a comparative study between a custom-built **Convolutional Neural Network (CNN)** and a **Vision Transformer (ViT)** for image classification on the CIFAR-10 dataset. The project focuses on modular software design, modern optimization techniques, and hardware-specific acceleration (Apple Silicon MPS).

## 🏗️ Model Architectures

### 1. SimpleCNN (Custom 4-Block Design)

A 4-layer convolutional architecture designed for spatial feature extraction:

* **Blocks**: Each block consists of `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d`.
* **Philosophy**: Prioritizes **Local Receptive Fields** and **Inductive Bias**, making it highly efficient for small-scale image datasets.
* **Regularization**: Integrated **Dropout** (0.5) in the classifier head to prevent overfitting.

### 2. Vision Transformer (ViT-Tiny)

Leverages the `timm` library to implement a SOTA Transformer-based classifier:

* **Mechanism**: Processes images as sequences of  patches.
* **Attention**: Utilizes **Self-Attention** to capture global dependencies across the entire image from the first layer.
* **Scalability**: Configurable image and patch sizes, demonstrating understanding of modern computer vision trends.

## ⚙️ Engineering & Training Pipeline

The training script (`train.py`) is built with industry-standard practices:

* **Hardware Acceleration**: Automatic device detection with support for **MPS (Metal Performance Shaders)** for optimized training on Apple Silicon.
* **Advanced Optimization**: Implements **AdamW** (Decoupled Weight Decay) and **CosineAnnealingLR** for superior convergence.
* **Early Stopping**: Monitors validation accuracy to prevent unnecessary compute and overfitting.
* **Data Handling**: Integration with **HuggingFace Datasets** and custom PyTorch `DataLoaders` with proper Train/Val/Test splitting.

## 📈 Performance & Metrics

The current baseline achieves **~80% Accuracy** on the test set.

### Key Trade-offs Identified:

* **CNN**: Faster inference and better performance on lower-resolution features.
* **ViT**: Superior at capturing global context but requires more computational resources and careful hyperparameter tuning.

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/RoiAviram/cifar10-cnn-vit-comparison.git

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Train the model**:
```bash
python train.py --model vit --epochs 10 --img-size 64

```



---

## 👨‍💻 Author

**Roi Aviram** | Electrical & Computer Engineering Student @ Ben-Gurion University.

*Specializing in Computer Vision, Deep Learning, and Real-time Systems.*

---
