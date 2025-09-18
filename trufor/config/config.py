# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
TruFor Configuration Management
"""

import os
from yacs.config import CfgNode as CN


class TruForConfig:
    """TruFor Configuration class for managing model and training parameters."""
    
    def __init__(self):
        self._C = CN()
        self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default configuration parameters."""
        _C = self._C
        
        _C.OUTPUT_DIR = 'weights'
        _C.LOG_DIR = 'log'
        _C.GPUS = (0,)
        _C.WORKERS = 4
        
        # Cudnn parameters
        _C.CUDNN = CN()
        _C.CUDNN.BENCHMARK = True
        _C.CUDNN.DETERMINISTIC = False
        _C.CUDNN.ENABLED = True
        
        # Model parameters
        _C.MODEL = CN()
        _C.MODEL.NAME = 'detconfcmx'
        _C.MODEL.PRETRAINED = ''
        _C.MODEL.MODS = ('RGB', 'NP++')
        _C.MODEL.EXTRA = CN(new_allowed=True)
        _C.MODEL.EXTRA.BACKBONE = 'mit_b2'
        _C.MODEL.EXTRA.DECODER = 'MLPDecoder'
        _C.MODEL.EXTRA.DECODER_EMBED_DIM = 512
        _C.MODEL.EXTRA.PREPRC = 'imagenet'
        _C.MODEL.EXTRA.BN_EPS = 0.001
        _C.MODEL.EXTRA.BN_MOMENTUM = 0.1
        _C.MODEL.EXTRA.DETECTION = 'confpool'
        _C.MODEL.EXTRA.MODULES = ['NP++', 'backbone', 'loc_head', 'conf_head', 'det_head']
        _C.MODEL.EXTRA.FIX_MODULES = ['NP++']
        _C.MODEL.EXTRA.NP_WEIGHTS = ''
        _C.MODEL.EXTRA.NP_OUT_CHANNELS = 1
        
        # Dataset parameters
        _C.DATASET = CN()
        _C.DATASET.ROOT = ''
        _C.DATASET.TRAIN = []
        _C.DATASET.VALID = []
        _C.DATASET.NUM_CLASSES = 2
        _C.DATASET.CLASS_WEIGHTS = [0.5, 2.5]
        
        # Training parameters
        _C.TRAIN = CN()
        _C.TRAIN.IMAGE_SIZE = [512, 512]
        _C.TRAIN.LR = 0.01
        _C.TRAIN.OPTIMIZER = 'sgd'
        _C.TRAIN.MOMENTUM = 0.9
        _C.TRAIN.WD = 0.0001
        _C.TRAIN.NESTEROV = False
        _C.TRAIN.IGNORE_LABEL = -1
        _C.TRAIN.BEGIN_EPOCH = 0
        _C.TRAIN.END_EPOCH = 100
        _C.TRAIN.BATCH_SIZE_PER_GPU = 18
        _C.TRAIN.SHUFFLE = True
        _C.TRAIN.NUM_SAMPLES = 0
        
        # Validation parameters
        _C.VALID = CN()
        _C.VALID.IMAGE_SIZE = None
        _C.VALID.FIRST_VALID = True
        _C.VALID.MAX_SIZE = None
        _C.VALID.BEST_KEY = 'avg_mIoU'
        
        # Testing parameters
        _C.TEST = CN()
        _C.TEST.MODEL_FILE = ''
    
    @property
    def config(self):
        """Get the configuration object."""
        return self._C
    
    def merge_from_file(self, config_file):
        """Merge configuration from a YAML file."""
        self._C.defrost()
        self._C.merge_from_file(config_file)
        self._C.freeze()
    
    def merge_from_list(self, config_list):
        """Merge configuration from a list."""
        self._C.defrost()
        self._C.merge_from_list(config_list)
        self._C.freeze()
    
    def update_config(self, config_updates=None):
        """Update configuration with custom parameters."""
        self._C.defrost()
        if config_updates:
            for key, value in config_updates.items():
                if hasattr(self._C, key):
                    setattr(self._C, key, value)
        self._C.freeze()


def update_config(cfg, args=None):
    """Utility function to update config with command line arguments."""
    cfg.defrost()
    
    if args and hasattr(args, 'experiment'):
        try:
            cfg.merge_from_file(f'lib/config/{args.experiment}.yaml')
        except:
            pass
    
    if args and hasattr(args, 'opts') and args.opts:
        cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg


# Create default config instance
config = TruForConfig().config
