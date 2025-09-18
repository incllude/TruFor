#!/usr/bin/env python3

"""
TruFor Basic Usage Example

This example demonstrates how to use the TruFor library for image forgery detection.
"""

import sys
import os

# Add the parent directory to path to import trufor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import trufor
    print("TruFor library imported successfully!")
    
    # Create a config
    config = trufor.TruForConfig()
    print("Configuration created")
    
    # You would normally do:
    # model = trufor.load_model('path/to/weights.pth')
    # results = trufor.predict_image(model, 'image.jpg')
    
    print("Basic setup complete!")
    print("To use TruFor for inference, you need:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download pretrained weights")
    print("3. Run: trufor.load_model('weights.pth')")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
