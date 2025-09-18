"""
TruFor: Image Forgery Detection and Localization Library

A Python library for detecting and localizing manipulations in images using deep learning.
Based on the TruFor paper and research code.
"""

__version__ = "1.0.0"
__author__ = "TruFor Team"
__email__ = "trufor@unina.it"

from .models import TruForModel
from .datasets import TruForDataset
from .config import TruForConfig
from .utils import load_model, predict_image, batch_predict, save_results, get_model_summary

__all__ = [
    "TruForModel",
    "TruForDataset", 
    "TruForConfig",
    "load_model",
    "predict_image",
    "batch_predict",
    "save_results", 
    "get_model_summary",
]
