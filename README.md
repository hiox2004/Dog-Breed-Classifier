<div align="center">

# 🐕 Dog Breed Classifier

### Deep Learning Multi-Class Image Classification Using Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-89.29%25-brightgreen.svg)](RESULTS.md)

*Achieving 89.29% validation accuracy on 120 dog breeds using InceptionV3 transfer learning*

[Features](#-features) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Demo](#-demo) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training-a-new-model)
  - [Evaluation](#evaluating-the-model)
  - [Prediction](#making-predictions)
- [Demo](#-demo)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

A production-ready deep learning pipeline for classifying 120 different dog breeds using transfer learning with **InceptionV3**. Built with TensorFlow/Keras, this project demonstrates best practices in computer vision including data augmentation, model evaluation, and reproducible results.

### Key Highlights

- **89.29%** validation accuracy (Top-1)
- **97.65%** Top-3 accuracy
- **99.17%** Top-5 accuracy
- 10,222 images across 120 breed classes
- Complete evaluation and reproducibility framework

---

## ✨ Features

- 🚀 **Transfer Learning** with InceptionV3 (ImageNet pre-trained)
- 📊 **Comprehensive Metrics** including Top-K accuracy, confusion matrix, per-class analysis
- 🔄 **Data Augmentation** pipeline for improved generalization
- 📈 **Training Visualization** with real-time metrics and callbacks
- 🎨 **Interactive Demos** for predictions with confidence visualization
- 📝 **Reproducible Results** with timestamped evaluation reports
- 🛠️ **Modular Architecture** for easy experimentation
- 💾 **Model Checkpointing** with automatic best model selection

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Validation Accuracy (Top-1)** | 89.29% |
| **Top-3 Accuracy** | 97.65% |
| **Top-5 Accuracy** | 99.17% |
| **Validation Loss** | 0.4527 |
| **Training Time** | ~3 hours (12 epochs) |
| **Model Size** | 87 MB |
| **Parameters** | 22.9M (21.8M frozen + 1.1M trainable) |

### Training Progress

| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|-------|-----------|---------|----------|-------|
| 1 | 35.43% | 59.90% | 1.6018 | Initial |
| 5 | 82.68% | 85.79% | 0.5371 | - |
| 8 | 88.95% | 88.41% | 0.4522 | - |
| **11** | **92.55%** | **89.29%** | **0.4527** | **Best** ✨ |
| 12 | 93.09% | 88.41% | 0.4687 | Early stop |

📄 **Detailed Results**: See [RESULTS.md](RESULTS.md) for complete evaluation reports and reproducibility instructions.

---

## 📁 Project Structure

```
DogBreedClassifier/
├── 📂 src/                          # Source code modules
│   ├── config.py                    # Configuration settings
│   ├── data.py                      # Data loading & augmentation
│   ├── model.py                     # Model architecture
│   └── predict.py                   # Prediction utilities
├── 📂 artifacts/                    # Trained models & outputs
│   ├── best_model.keras            # Best model checkpoint (89.29%)
│   ├── classes.json                # Class names mapping
│   ├── confusion_matrix.png        # Confusion matrix visualization
│   └── class_distribution.png      # Dataset distribution chart
├── 📂 evaluation_results/           # Evaluation reports
│   ├── evaluation_report_*.json    # Detailed metrics (JSON)
│   ├── evaluation_summary_*.txt    # Human-readable summary
│   └── confusion_matrix_*.png      # Timestamped confusion matrix
├── 📂 data/                         # Dataset (not included in repo)
│   └── train/                       # Training images
├── train.py                         # Main training script
├── evaluate_model.py                # Model evaluation script
├── predict_demo.py                  # Interactive prediction demo
├── finetune_from_checkpoint.py     # Fine-tuning script
├── requirements.txt                 # Python dependencies
├── RESULTS.md                       # Detailed results documentation
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- Windows PowerShell (or Bash for Linux/Mac)
- ~2GB disk space for dataset
- GPU recommended (but not required)

### Setup

1. **Clone the repository**
   ```powershell
   git clone https://github.com/hiox2004/Dog-Breed-Classifier.git
   cd Dog-Breed-Classifier
   ```

2. **Create virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   # source .venv/bin/activate   # Linux/Mac
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download from [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data)
   - Extract to `data/train/` directory
   - Place `labels.csv` in project root

---

## 💻 Usage

### Training a New Model

**Basic training** (uses InceptionV3 by default):
```powershell
python train.py
```

**Advanced training** with custom parameters:
```powershell
python train.py --base_model InceptionV3 --epochs 50 --batch_size 32 --lr 0.001
```

**Available options:**
- `--base_model`: Choose architecture (`InceptionV3`, `MobileNetV2`)
- `--epochs`: Maximum training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Initial learning rate (default: 0.001)
- `--image_size`: Image dimensions (default: 299,299 for InceptionV3)

### Evaluating the Model

**Run comprehensive evaluation**:
```powershell
python evaluate_model.py --model artifacts/best_model.keras
```

This generates:
- 📊 `evaluation_report_*.json` - Detailed metrics
- 📝 `evaluation_summary_*.txt` - Human-readable report
- 🎨 `confusion_matrix_*.png` - Confusion matrix visualization

### Making Predictions

**CLI prediction** (quick):
```powershell
python -m src.predict --image path\to\dog.jpg
```

**Interactive demo** (with visualization):
```powershell
python predict_demo.py path\to\dog.jpg
```

Example output:
```
Top 3 Predictions:
1. Golden Retriever    (87.3%)
2. Labrador Retriever  (8.9%)
3. Flat-Coated Retriever (2.1%)
```

---

## 🎨 Demo

### Prediction Examples

Run the interactive demo to see predictions with confidence scores and visualizations:

```powershell
python predict_demo.py path\to\your\dog_image.jpg
```

The demo will:
- Display the input image
- Show top-3 breed predictions with confidence bars
- Save output as `prediction_output.png`

---

## 🏗️ Model Architecture

### InceptionV3 Transfer Learning

```
Input (299x299x3)
    ↓
InceptionV3 Base (frozen)
  - 21.8M parameters
  - ImageNet pre-trained
    ↓
Global Average Pooling
    ↓
Dense (512, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (120, Softmax)
  - 1.1M trainable parameters
    ↓
Output (120 classes)
```

### Key Features

- **Frozen Base**: Pre-trained InceptionV3 weights remain fixed
- **Custom Head**: Trainable classification layers for 120 breeds
- **Regularization**: Dropout to prevent overfitting
- **Optimization**: Adam optimizer with learning rate scheduling
- **Callbacks**: Early stopping, LR reduction, model checkpointing

---

## 📊 Dataset

### Stanford Dogs Dataset (Kaggle)

- **Total Images**: 10,222
- **Classes**: 120 dog breeds
- **Split**: 80% train (8,177) / 20% validation (2,045)
- **Image Format**: JPEG, variable sizes
- **Preprocessing**: Resized to 299×299, normalized [0,1]

### Data Augmentation

Training augmentation pipeline:
- Random rotation (±15°)
- Random translation (10%)
- Random zoom (10%)
- Random horizontal flip
- On-the-fly generation (no storage overhead)

---

## 📚 Documentation

- **[RESULTS.md](RESULTS.md)** - Detailed evaluation results and reproducibility guide
- **[requirements.txt](requirements.txt)** - Python package dependencies
- **Code Documentation** - Inline docstrings and comments throughout

### Key Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Train model from scratch |
| `evaluate_model.py` | Evaluate and generate reports |
| `predict_demo.py` | Interactive prediction demo |
| `finetune_from_checkpoint.py` | Fine-tune pre-trained model |
| `src/data.py` | Data loading and augmentation |
| `src/model.py` | Model architecture definitions |
| `src/predict.py` | Prediction utilities |

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🔧 Submit pull requests

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) via Kaggle
- **Pre-trained Model**: InceptionV3 trained on ImageNet
- **Framework**: TensorFlow/Keras team
- **Inspiration**: Transfer learning best practices from the deep learning community

---

## 📧 Contact

**Your Name** - [@hiox2004](https://github.com/hiox2004)

**Project Link**: [https://github.com/hiox2004/Dog-Breed-Classifier](https://github.com/hiox2004/Dog-Breed-Classifier)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ and 🐕

</div>
