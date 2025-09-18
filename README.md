# TruFor: Image Forgery Detection and Localization Library

TruFor is a Python library for detecting and localizing manipulations in images using deep learning. It implements the TruFor model from the paper "TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization" by Guillaro et al.

## Features

- **Image Forgery Detection**: Classify images as authentic or manipulated
- **Manipulation Localization**: Identify specific regions that have been altered
- **Confidence Estimation**: Get confidence scores for predictions
- **Multi-modal Analysis**: Combines RGB features with Noiseprint++ analysis
- **Easy-to-use API**: Simple interface for both research and practical applications

## Installation

Install TruFor using pip:

```bash
pip install trufor
```

For development installation:

```bash
git clone https://github.com/grip-unina/TruFor
cd TruFor
pip install -e .
```

## Quick Start

### Basic Usage

```python
import trufor

# Load pretrained model
model = trufor.load_model('path/to/weights.pth')

# Predict on a single image
results = trufor.predict_image(model, 'image.jpg')

print(f"Authenticity: {results['authenticity']}")
print(f"Detection Score: {results['detection_score']}")

# Access localization map
localization_map = results['localization']
```

### Advanced Usage

```python
from trufor import TruForModel, TruForConfig
from trufor.datasets import TestDataset
import torch

# Custom configuration
config = TruForConfig()
config.merge_from_list(['MODEL.EXTRA.BACKBONE', 'mit_b2'])

# Create model
model = TruForModel(config)
model.load_pretrained('weights.pth')

# Batch prediction
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = trufor.batch_predict(model, image_paths, batch_size=2)

# Save results with visualizations
for i, result in enumerate(results):
    trufor.save_results(
        result, 
        output_dir='outputs/',
        filename_prefix=f'image_{i}'
    )
```

### Dataset Usage

```python
from trufor.datasets import TruForDataset, TestDataset
from torch.utils.data import DataLoader

# Test dataset for inference
test_dataset = TestDataset('path/to/images/')
test_loader = DataLoader(test_dataset, batch_size=1)

# Training dataset (requires image list)
train_dataset = TruForDataset(
    image_list=['img1.jpg', 'img2.jpg'],
    crop_size=512,
    mode='train'
)
```

## Model Architecture

TruFor combines:

1. **RGB Branch**: Processes natural image features using SegFormer backbone
2. **Noiseprint++ Branch**: Analyzes camera sensor noise patterns  
3. **Cross-Modal Exchange (CMX)**: Fuses RGB and noise features
4. **Multi-task Heads**: 
   - Localization head for pixel-level manipulation detection
   - Confidence head for reliability estimation
   - Detection head for image-level classification

## Pretrained Models

Download pretrained weights from:
- [TruFor Weights](https://github.com/grip-unina/TruFor/releases)

Place weights in your project directory and load with:

```python
model = trufor.load_model('trufor_weights.pth')
```

## API Reference

### Core Functions

- `trufor.load_model(model_path, config=None, device='auto')` - Load pretrained model
- `trufor.predict_image(model, image_path, device='auto')` - Single image prediction
- `trufor.batch_predict(model, image_paths, batch_size=1)` - Batch prediction

### Classes

- `TruForModel` - Main model class
- `TruForConfig` - Configuration management
- `TruForDataset` - Base dataset class
- `TestDataset` - Simple test dataset

### Utilities

- `preprocess_image()` - Image preprocessing
- `save_results()` - Save predictions to files  
- `get_model_summary()` - Model information

## Command Line Interface

TruFor provides CLI tools for quick testing:

```bash
# Predict single image
trufor-predict --input image.jpg --output results/ --model weights.pth

# Get model information
trufor-info --model weights.pth
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- OpenCV >= 4.5.0
- Additional dependencies in `requirements.txt`

## Citation

If you use TruFor in your research, please cite:

```bibtex
@article{guillaro2023trufor,
  title={TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization},
  author={Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## License

This project is licensed under a custom license for non-profit use only. See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments
 
- Image Processing Research Group of University Federico II of Naples (GRIP-UNINA)
- Original TruFor paper authors
- SegFormer and Noiseprint++ implementations

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Support

- 📖 [Documentation](https://grip-unina.github.io/TruFor/)
- 🐛 [Issue Tracker](https://github.com/grip-unina/TruFor/issues)  
- 📧 Contact: trufor@unina.it