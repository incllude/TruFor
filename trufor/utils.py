# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.

"""
TruFor Utilities: Helper functions for model loading and inference.
"""

import os
import torch
import numpy as np
from typing import Union, Dict, Optional
from PIL import Image

from .models import TruForModel
from .config import TruForConfig


def load_model(model_path: str, 
               config: Optional[TruForConfig] = None,
               device: str = 'auto') -> TruForModel:
    """
    Load a pretrained TruFor model.
    
    Args:
        model_path: Path to the pretrained model file
        config: Optional configuration object
        device: Device to load model on ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        Loaded TruFor model
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if config is None:
        config = TruForConfig()
    
    # Create model
    model = TruForModel(config)
    
    # Load pretrained weights
    model.load_pretrained(model_path)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model


def predict_image(model: TruForModel,
                  image_path: str,
                  device: str = 'auto',
                  return_visualization: bool = False) -> Dict:
    """
    Predict image authenticity and localize manipulations.
    
    Args:
        model: TruFor model
        image_path: Path to the image file
        device: Device for inference
        return_visualization: Whether to return visualization images
        
    Returns:
        Dictionary with prediction results
    """
    if device == 'auto':
        device = next(model.parameters()).device
    
    # Get prediction
    results = model.predict(image_path, device=str(device))
    
    if return_visualization:
        # Add visualization images
        localization_map = results['localization'].squeeze().numpy()
        
        # Create heatmap visualization
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Normalize for visualization
        heatmap = cm.jet(localization_map)[:, :, :3]  # Remove alpha channel
        
        results['localization_heatmap'] = (heatmap * 255).astype(np.uint8)
        
        # Load original image for overlay
        original_image = np.array(Image.open(image_path).convert('RGB'))
        
        # Create overlay
        alpha = 0.4
        overlay = (alpha * heatmap * 255 + (1 - alpha) * original_image).astype(np.uint8)
        results['overlay'] = overlay
    
    return results


def preprocess_image(image: Union[str, np.ndarray, Image.Image],
                     target_size: Optional[tuple] = None) -> torch.Tensor:
    """
    Preprocess image for TruFor inference.
    
    Args:
        image: Input image (path, numpy array, or PIL Image)
        target_size: Optional target size (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize if needed
    if target_size:
        image = image.resize((target_size[1], target_size[0]))
    
    # Convert to numpy and normalize
    image_np = np.array(image) / 255.0
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def save_results(results: Dict, 
                 output_dir: str,
                 filename_prefix: str = "result"):
    """
    Save prediction results to files.
    
    Args:
        results: Prediction results dictionary
        output_dir: Output directory
        filename_prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save localization map
    if 'localization' in results:
        loc_map = results['localization'].squeeze().numpy()
        loc_image = Image.fromarray((loc_map * 255).astype(np.uint8))
        loc_image.save(os.path.join(output_dir, f"{filename_prefix}_localization.png"))
    
    # Save confidence map
    if 'confidence' in results and results['confidence'] is not None:
        conf_map = results['confidence'].squeeze().numpy()
        conf_image = Image.fromarray((conf_map * 255).astype(np.uint8))
        conf_image.save(os.path.join(output_dir, f"{filename_prefix}_confidence.png"))
    
    # Save heatmap visualization if available
    if 'localization_heatmap' in results:
        heatmap_image = Image.fromarray(results['localization_heatmap'])
        heatmap_image.save(os.path.join(output_dir, f"{filename_prefix}_heatmap.png"))
    
    # Save overlay if available
    if 'overlay' in results:
        overlay_image = Image.fromarray(results['overlay'])
        overlay_image.save(os.path.join(output_dir, f"{filename_prefix}_overlay.png"))
    
    # Save text results
    with open(os.path.join(output_dir, f"{filename_prefix}_results.txt"), 'w') as f:
        f.write(f"Authenticity: {results['authenticity']}\n")
        if 'detection_score' in results and results['detection_score'] is not None:
            f.write(f"Detection Score: {results['detection_score'].item():.4f}\n")


def batch_predict(model: TruForModel,
                  image_paths: list,
                  batch_size: int = 1,
                  device: str = 'auto') -> list:
    """
    Predict multiple images in batches.
    
    Args:
        model: TruFor model
        image_paths: List of image paths
        batch_size: Batch size for processing
        device: Device for inference
        
    Returns:
        List of prediction results
    """
    if device == 'auto':
        device = next(model.parameters()).device
    
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_results = []
        
        for path in batch_paths:
            result = model.predict(path, device=str(device))
            result['image_path'] = path
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results


def get_model_summary(model: TruForModel) -> Dict:
    """
    Get model summary and information.
    
    Args:
        model: TruFor model
        
    Returns:
        Model information dictionary
    """
    return model.get_model_info()
