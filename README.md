# Swajay CV Toolkit 🚀

[![PyPI version](https://badge.fury.io/py/swajay-cv-toolkit.svg)](https://badge.fury.io/py/swajay-cv-toolkit)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/swajay-cv-toolkit)](https://pepy.tech/project/swajay-cv-toolkit)

**Advanced Computer Vision Toolkit for State-of-the-Art Image Classification**

A comprehensive, production-ready toolkit featuring cutting-edge loss functions, model architectures, augmentation strategies, and training pipelines. Designed for researchers, practitioners, and Kaggle competitors who want to achieve state-of-the-art results with minimal code.

## 🎯 Key Features

### 🔥 **Advanced Loss Functions**
- **Focal Loss** - Handle class imbalance effectively
- **Label Smoothing Cross Entropy** - Improve generalization  
- **Polynomial Loss** - Enhanced convergence properties
- **Mixed Loss** - Optimal combination of multiple loss functions
- **Automatic class weight calculation** for imbalanced datasets

### 🏗️ **Universal Model Architectures**  
- **Auto-adaptive models** - Automatically adjust to any number of classes
- **Multiple architectures** - ConvNeXt, EfficientNet, ResNet, Vision Transformers
- **Smart classifier heads** - Optimal architecture for your dataset size
- **Ensemble support** - Combine multiple models for better performance

### 🎭 **Professional Augmentations**
- **Competition-grade augmentations** - Albumentations-based pipeline
- **MixUp & CutMix** - Advanced mixing strategies
- **Test-Time Augmentation (TTA)** - Boost inference accuracy
- **4 intensity levels** - From lightweight to competition-grade

### 🏋️ **Advanced Training Pipeline**
- **Mixed precision training** - 2x faster with lower memory usage
- **Differential learning rates** - Optimal rates for backbone vs classifier
- **Smart early stopping** - Prevent overfitting automatically
- **Comprehensive metrics** - Track everything that matters

## 🚀 Quick Start

### Installation

```bash
pip install swajay-cv-toolkit
```

### Basic Usage

```python
import torch
from swajay_cv_toolkit import create_model, get_loss_function, AdvancedTrainer
from torch.utils.data import DataLoader

# Create model for any number of classes
model = create_model('convnext_large', num_classes=10)

# Get advanced loss function
criterion = get_loss_function('mixed')  # Combines focal + label smoothing + poly

# Create optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Advanced trainer with all the bells and whistles
trainer = AdvancedTrainer(
    model=model,
    criterion=criterion, 
    optimizer=optimizer,
    mixed_precision=True,
    gradient_clip_norm=1.0
)

# Train with automatic early stopping
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader, 
    epochs=50,
    early_stopping_patience=10
)
```

### Complete Pipeline Example

```python
from swajay_cv_toolkit import *

# 1. Setup
seed_everything(42)
device = setup_device()

# 2. Data augmentation
aug_config = get_augmentation_preset('competition', image_size=224)
train_dataset = AdvancedDataset(raw_dataset, aug_config['train_transform'])

# 3. Model creation  
model = create_model('convnext_large', num_classes=20).to(device)

# 4. Advanced training components
criterion = get_loss_function('mixed')
optimizer = create_optimizer(model, 'adamw', lr=1e-3, differential_lr=True)
scheduler = create_scheduler(optimizer, 'cosine', total_steps=1000)

# 5. Train with MixUp/CutMix
mixup_cutmix = MixupCutmix(mixup_alpha=0.2, cutmix_alpha=1.0)
trainer = AdvancedTrainer(model, criterion, optimizer, scheduler)
history = trainer.fit(train_loader, val_loader, epochs=30, mixup_cutmix=mixup_cutmix)

# 6. Test-Time Augmentation for final predictions
tta_predictor = TTAPredictor(model, device)
predictions = tta_predictor.predict_with_tta(test_dataset, aug_config['tta_transforms'])
```

## 🎯 Supported Use Cases

### ✅ **Works on ANY Image Classification Task**
- **Medical Imaging** - X-rays, MRIs, pathology slides
- **Satellite Imagery** - Land use, crop monitoring, disaster assessment  
- **Manufacturing** - Quality control, defect detection
- **Retail & E-commerce** - Product categorization, visual search
- **Security** - Face recognition, object identification
- **Agriculture** - Plant disease detection, crop classification
- **Wildlife & Conservation** - Species identification, monitoring
- **Food & Nutrition** - Recipe classification, nutritional analysis
- **Academic Research** - Any custom image classification dataset

### 📊 **Dataset Requirements**
Just organize your data like this:
```
your_dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...
```

That's it! The toolkit handles everything else automatically.



## 📚 Comprehensive Examples

### Example 1: CIFAR-10 Classification
```python
import torchvision.datasets as datasets
from swajay_cv_toolkit import *

# Load CIFAR-10
train_data = datasets.CIFAR10(root='./data', train=True, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, download=True)

# Quick setup with presets
config = {
    'model': 'efficientnet_b4',
    'num_classes': 10,
    'image_size': 224
}

# One-line training
model, history = quick_train(
    train_data, test_data, 
    preset='competition',
    **config
)
```

### Example 2: Custom Medical Dataset
```python
# Your medical imaging dataset
train_data = datasets.ImageFolder('./medical_images/train')
test_data = datasets.ImageFolder('./medical_images/test')

# Specialized setup for medical images
model = create_model('convnext_large', num_classes=len(train_data.classes))
criterion = get_loss_function('focal', alpha=2, gamma=2)  # Good for imbalanced medical data

# Conservative augmentations for medical data
aug_config = get_augmentation_preset('standard', image_size=384)

# Train with higher resolution for medical precision
trainer = AdvancedTrainer(model, criterion, optimizer)
history = trainer.fit(train_loader, val_loader, epochs=100)
```

### Example 3: Ensemble for Maximum Accuracy
```python
# Create ensemble of different architectures
models = [
    create_model('convnext_large', num_classes=20),
    create_model('efficientnet_b7', num_classes=20),
    create_model('vit_large_patch16_224', num_classes=20)
]

ensemble = EnsembleModel(models, weights=[0.4, 0.35, 0.25])

# Train ensemble or individual models
for i, model in enumerate(models):
    print(f"Training model {i+1}/3...")
    trainer = AdvancedTrainer(model, criterion, optimizer)
    trainer.fit(train_loader, val_loader, epochs=30)
```

## 🔧 Advanced Features

### 🎯 **Automatic Configuration**
- **Auto-detect classes** - No need to count manually
- **Auto-size networks** - Optimal architecture for your data size
- **Auto-balance classes** - Handle imbalanced datasets automatically
- **Auto-tune hyperparameters** - Smart defaults that work

### 🔬 **Research-Grade Components**
All components are based on peer-reviewed research:

- **ConvNeXt**: Liu et al., 2022 - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- **Focal Loss**: Lin et al., 2017 - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **Label Smoothing**: Szegedy et al., 2016 - [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567)  
- **MixUp**: Zhang et al., 2017 - [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- **Test-Time Augmentation**: He et al., 2015 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **AdamW**: Loshchilov & Hutter, 2017 - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

### 🚀 **Production Ready**
- **Memory efficient** - Automatic mixed precision
- **GPU optimized** - Efficient data loading and processing
- **Reproducible** - Comprehensive seed management
- **Robust** - Extensive error handling and validation
- **Scalable** - Works from laptop to multi-GPU setups

## 🎮 Presets for Every Use Case

### `'lightweight'` - Fast Development
- EfficientNet-B2, basic augmentations, 20 epochs
- Perfect for prototyping and quick experiments

### `'standard'` - Balanced Performance  
- ConvNeXt-Base, medium augmentations, 30 epochs
- Great balance of speed and accuracy

### `'competition'` - Maximum Accuracy
- ConvNeXt-Large, aggressive augmentations, TTA, 40 epochs  
- For when you need the absolute best results

### `'research'` - Experimental Features
- Latest architectures, experimental techniques
- For pushing the boundaries

## 📖 Documentation

### 🚀 **Quick References**
- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md) 
- [API Reference](docs/api_reference.md)
- [Example Gallery](examples/)

### 📋 **Detailed Guides**
- [Loss Functions Guide](docs/losses.md) - When to use which loss
- [Model Architecture Guide](docs/models.md) - Choosing the right model
- [Augmentation Guide](docs/augmentations.md) - Augmentation strategies
- [Training Guide](docs/training.md) - Advanced training techniques
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest tests/`
5. **Format code**: `black swajay_cv_toolkit/`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### 🐛 **Bug Reports**
Found a bug? [Open an issue](https://github.com/swajayresources/swajay-cv-toolkit/issues) with:
- Python version and OS
- Full error traceback
- Minimal code to reproduce
- Dataset information (if relevant)

### 💡 **Feature Requests**
Have an idea? [Request a feature](https://github.com/swajayresources/swajay-cv-toolkit/issues) with:
- Use case description
- Expected behavior
- Example code (if possible)





## 🛠️ Technical Requirements

### Minimum Requirements
- **Python**: 3.8+
- **PyTorch**: 1.12.0+
- **CUDA**: Optional but recommended
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 2GB for dependencies

### Recommended Setup
- **GPU**: RTX 3080+ or Tesla V100+
- **RAM**: 32GB+
- **Python**: 3.10+
- **CUDA**: 11.7+

### Dependencies
All dependencies are automatically installed:
```
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.0
albumentations>=1.3.0
opencv-python-headless>=4.6.0
numpy>=1.21.0
scikit-learn>=1.1.0
pandas>=1.4.0
```


### 💬 **Get Help**
- **Documentation**: [swajay-cv-toolkit.readthedocs.io](https://swajay-cv-toolkit.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/swajayresources/swajay-cv-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/swajayresources/swajay-cv-toolkit/discussions)
- **Email**: swajay@example.com

### 🌟 **Show Your Support**
If this toolkit helps you:
- ⭐ **Star the repository**
- 🐦 **Tweet about it** 
- 📝 **Write a blog post**
- 🗣️ **Tell your colleagues**
- 💖 **Sponsor the project**

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{swajay_cv_toolkit,
  title={Swajay CV Toolkit: Advanced Computer Vision for Image Classification},
  author={Swajay},
  year={2024},
  url={https://github.com/swajayresources/swajay-cv-toolkit},
  version={1.0.0}
}
```

## 🙏 Acknowledgments

Special thanks to:
- **PyTorch Team** - For the amazing framework
- **TIMM contributors** - For the model implementations
- **Albumentations team** - For the augmentation library
- **Research community** - For the foundational papers
- **Beta testers** - For feedback and bug reports
- **Contributors** - For making this project better

---

<div align="center">

**Made with ❤️ by Swajay**

[⭐ Star on GitHub](https://github.com/swajayresources/swajay-cv-toolkit) • [📖 Documentation](https://swajay-cv-toolkit.readthedocs.io/) • [🐛 Report Bug](https://github.com/swajayresources/swajay-cv-toolkit/issues) • [💡 Request Feature](https://github.com/swajayresources/swajay-cv-toolkit/issues)

</div>