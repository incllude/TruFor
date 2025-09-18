# TruFor Examples

This directory contains example scripts showing how to use the TruFor library.

## Prerequisites

Before running the examples, make sure you have:

1. Installed the TruFor library: `pip install trufor` (or `pip install -e .` for development)
2. Installed all dependencies: `pip install -r requirements.txt`
3. Downloaded the pretrained model weights

## Examples

### basic_usage.py
Demonstrates basic library import and configuration setup.

```bash
python basic_usage.py
```

### Advanced Usage (when dependencies are installed)

```python
import trufor

# Load model
model = trufor.load_model('path/to/trufor_weights.pth')

# Single image prediction
results = trufor.predict_image(model, 'test_image.jpg')
print(f"Authenticity: {results['authenticity']}")

# Batch prediction
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_results = trufor.batch_predict(model, image_paths)

# Save results with visualizations
for i, result in enumerate(batch_results):
    trufor.save_results(result, 'output/', f'result_{i}')
```

## CLI Usage

After installation, you can use the command-line interface:

```bash
# Single image
trufor-predict --input image.jpg --model weights.pth --output results/

# Multiple images
trufor-predict --input images/ --model weights.pth --output results/ --visualize

# Model information
trufor-info --model weights.pth
```
