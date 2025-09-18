# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.

"""
TruFor Dataset classes for training and validation.
"""

import os
import random
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class TruForDataset(Dataset):
    """
    Base TruFor dataset class for handling image forensics data.
    """
    
    def __init__(self, 
                 image_list: List[str],
                 crop_size: Union[int, Tuple[int, int]] = 512,
                 mode: str = "train",
                 max_dim: Optional[int] = None,
                 transform=None):
        """
        Initialize TruFor dataset.
        
        Args:
            image_list: List of image file paths
            crop_size: Size for cropping images (int or tuple)
            mode: Dataset mode ("train" or "val")
            max_dim: Maximum dimension for resizing
            transform: Optional image transformations
        """
        self.image_list = image_list
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.mode = mode
        self.max_dim = max_dim
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        Get item from dataset.
        
        Args:
            index: Item index
            
        Returns:
            Tuple of (image_tensor, image_path)
        """
        image_path = self.image_list[index]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Resize if max_dim specified
        if self.max_dim:
            h, w = image.shape[:2]
            if max(h, w) > self.max_dim:
                scale = self.max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = np.array(Image.fromarray(image).resize((new_w, new_h)))
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image, image_path
    
    def shuffle(self):
        """Shuffle the dataset."""
        random.shuffle(self.image_list)
    
    def get_filename(self, index: int) -> str:
        """Get filename at given index."""
        return self.image_list[index]


class TestDataset(TruForDataset):
    """
    Simple test dataset for inference.
    """
    
    def __init__(self, image_paths: Union[str, List[str]], **kwargs):
        """
        Initialize test dataset.
        
        Args:
            image_paths: Single image path, directory path, or list of paths
            **kwargs: Additional arguments passed to parent class
        """
        if isinstance(image_paths, str):
            if os.path.isfile(image_paths):
                # Single file
                image_list = [image_paths]
            elif os.path.isdir(image_paths):
                # Directory - get all image files
                image_list = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    image_list.extend([
                        os.path.join(image_paths, f) 
                        for f in os.listdir(image_paths) 
                        if f.lower().endswith(ext)
                    ])
            else:
                raise ValueError(f"Invalid path: {image_paths}")
        else:
            image_list = image_paths
        
        super().__init__(image_list, mode="test", **kwargs)


class ForensicsDataset(Dataset):
    """
    Advanced forensics dataset with support for masks and labels.
    """
    
    def __init__(self,
                 data_list: List[Tuple[str, Optional[str], int]],
                 crop_size: Union[int, Tuple[int, int]] = 512,
                 mode: str = "train",
                 transform=None):
        """
        Initialize forensics dataset.
        
        Args:
            data_list: List of tuples (image_path, mask_path, label)
            crop_size: Size for cropping
            mode: Dataset mode
            transform: Image transformations
        """
        self.data_list = data_list
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.mode = mode
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int) -> dict:
        """
        Get item with image, mask, and label.
        
        Returns:
            Dictionary with keys: 'image', 'mask', 'label', 'path'
        """
        image_path, mask_path, label = self.data_list[index]
        
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load mask if available
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                image = self.transform(image=image)['image']
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        result = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': image_path
        }
        
        if mask is not None:
            result['mask'] = torch.from_numpy(mask).float() / 255.0
        
        return result
