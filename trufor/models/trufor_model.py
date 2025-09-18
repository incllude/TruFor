# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.

"""
TruFor Model: Main model class for image forgery detection and localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from typing import Optional, Tuple, Union

from .cmx.builder_np_conf import EncoderDecoder
from ..config import TruForConfig


class TruForModel(nn.Module):
    """
    TruFor model for image forgery detection and localization.
    
    This model combines RGB image analysis with Noiseprint++ features
    to detect and localize image manipulations.
    """
    
    def __init__(self, config: Optional[TruForConfig] = None, **kwargs):
        """
        Initialize TruFor model.
        
        Args:
            config: TruFor configuration object
            **kwargs: Additional configuration parameters
        """
        super(TruForModel, self).__init__()
        
        if config is None:
            config = TruForConfig()
        
        self.config = config.config if hasattr(config, 'config') else config
        self.model = EncoderDecoder(cfg=self.config)
    
    def forward(self, rgb: torch.Tensor, save_np: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the TruFor model.
        
        Args:
            rgb: Input RGB image tensor [B, 3, H, W] in range [0, 1]
            save_np: Whether to return Noiseprint++ features
            
        Returns:
            Tuple containing:
            - out: Localization output [B, 2, H, W]
            - conf: Confidence map [B, 1, H, W] (if enabled)
            - det: Detection score [B, 1] (if enabled)
            - modal_x: Noiseprint++ features (if save_np=True)
        """
        return self.model(rgb, save_np=save_np)
    
    def predict(self, image: Union[torch.Tensor, str], device: str = 'cpu') -> dict:
        """
        Predict forgery for a single image.
        
        Args:
            image: Input image as tensor or file path
            device: Device to run inference on
            
        Returns:
            Dictionary containing predictions:
            - localization: Localization map
            - confidence: Confidence map (if available)
            - detection_score: Detection score (if available)
            - authenticity: Predicted authenticity (authentic/forged)
        """
        self.eval()
        self.to(device)
        
        if isinstance(image, str):
            from PIL import Image
            import numpy as np
            img = Image.open(image).convert('RGB')
            img = np.array(img) / 255.0
            image = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(device)
        
        with torch.no_grad():
            out, conf, det, _ = self.forward(image)
            
            # Process outputs
            localization = F.softmax(out, dim=1)[:, 1:2]  # Manipulation probability
            
            results = {
                'localization': localization.cpu(),
                'confidence': conf.cpu() if conf is not None else None,
                'detection_score': torch.sigmoid(det).cpu() if det is not None else None,
            }
            
            # Determine authenticity
            if det is not None:
                authenticity = 'forged' if torch.sigmoid(det).item() > 0.5 else 'authentic'
            else:
                # Use localization map if detection head not available
                manip_ratio = (localization > 0.5).float().mean().item()
                authenticity = 'forged' if manip_ratio > 0.1 else 'authentic'
            
            results['authenticity'] = authenticity
            
        return results
    
    def load_pretrained(self, model_path: str, strict: bool = True):
        """
        Load pretrained weights.
        
        Args:
            model_path: Path to pretrained model file
            strict: Whether to strictly enforce key matching
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logging.info(f"Loading pretrained model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=strict)
        logging.info("Pretrained model loaded successfully")
    
    def get_model_info(self) -> dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TruFor',
            'backbone': self.config.MODEL.EXTRA.BACKBONE,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'modalities': self.config.MODEL.MODS,
            'modules': self.config.MODEL.EXTRA.MODULES,
        }
