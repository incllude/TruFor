#!/usr/bin/env python3

"""
TruFor Command Line Interface

Provides command-line tools for image forgery detection.
"""

import argparse
import os
import sys
from typing import List

import torch

from . import load_model, predict_image, batch_predict, save_results


def predict_command():
    """Command line interface for TruFor prediction."""
    parser = argparse.ArgumentParser(
        description="TruFor Image Forgery Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trufor-predict --input image.jpg --model weights.pth
  trufor-predict --input images/ --output results/ --model weights.pth --batch-size 4
  trufor-predict --input "*.jpg" --model weights.pth --device cuda --visualize
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image file, directory, or glob pattern'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./trufor_output',
        help='Output directory for results (default: ./trufor_output)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to TruFor model weights file'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run inference on (default: auto)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size for processing multiple images (default: 1)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization images (heatmaps, overlays)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Detection threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output messages'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get input files
    input_files = get_input_files(args.input)
    if not input_files:
        print(f"Error: No image files found in: {args.input}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(input_files)} image(s) to process")
        print(f"Loading model from: {args.model}")
    
    # Load model
    try:
        model = load_model(args.model, device=args.device)
        if not args.quiet:
            print(f"Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process images
    try:
        if len(input_files) == 1:
            # Single image
            result = predict_image(
                model, 
                input_files[0], 
                device=args.device,
                return_visualization=args.visualize
            )
            
            filename = os.path.splitext(os.path.basename(input_files[0]))[0]
            save_results(result, args.output, filename)
            
            if not args.quiet:
                print(f"Result: {result['authenticity']}")
                if result['detection_score'] is not None:
                    print(f"Detection score: {result['detection_score'].item():.4f}")
        
        else:
            # Multiple images
            results = batch_predict(model, input_files, args.batch_size, args.device)
            
            for i, result in enumerate(results):
                filename = os.path.splitext(os.path.basename(input_files[i]))[0]
                save_results(result, args.output, filename)
                
                if not args.quiet:
                    print(f"{filename}: {result['authenticity']}")
        
        if not args.quiet:
            print(f"Results saved to: {args.output}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


def info_command():
    """Command line interface for model information."""
    parser = argparse.ArgumentParser(
        description="Get TruFor model information"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to TruFor model weights file'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    try:
        model = load_model(args.model, device='cpu')
        info = model.get_model_info()
        
        print("TruFor Model Information:")
        print("-" * 30)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def get_input_files(input_path: str) -> List[str]:
    """Get list of input image files from path or pattern."""
    import glob
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if os.path.isfile(input_path):
        # Single file
        return [input_path]
    
    elif os.path.isdir(input_path):
        # Directory - find all image files
        files = []
        for ext in image_extensions:
            pattern = os.path.join(input_path, f"*{ext}")
            files.extend(glob.glob(pattern))
            pattern = os.path.join(input_path, f"*{ext.upper()}")
            files.extend(glob.glob(pattern))
        return sorted(list(set(files)))
    
    else:
        # Glob pattern
        files = glob.glob(input_path)
        # Filter for image files
        return [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]


if __name__ == "__main__":
    predict_command()
