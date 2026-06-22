
#Wheat Spike Phenotyping Configuration for RESNET101-U-Net-scse architecture and YOLO11X-seg
from pathlib import Path
import torch


# Pretrained weights
PRETRAINED_WEIGHTS_DIR = PROJECT_ROOT / "pretrained_weights"
RESNET101_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIR / "resnet101-63fe2227.pth"
YOLO11X_SEG_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIR / "yolo11x-seg.pt"

ORIGINAL_WIDTH = 8192
ORIGINAL_HEIGHT = 5464
REFERENCE_MARKER_DIAMETER_MM = 14.0

# SPIKE SEGMENTATION CONFIG (ResNet101-UNet-scse)
SPIKE_CONFIG = {
    'image_size': 2560,
    'num_classes': 3,
    'batch_size': 2,
    'epochs': 800,
    'initial_lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'focal_alpha': 0.5,
    'focal_gamma': 2.0,
    'early_stopping_patience': 80,
    'encoder_name': 'resnet101',
    'encoder_weights': None,
    'encoder_weights_path': str(RESNET101_WEIGHTS_PATH),
    'decoder_attention': 'scse',
    'class_names': ['background', 'Head', 'Stem'],
    'val_split': 0.15,
}

# SPIKELET SEGMENTATION CONFIG (YOLO)
SPIKELET_CONFIG = {
    # Model
    'model_name': 'yolo11x-seg',
    'weights_path': str(YOLO11X_SEG_WEIGHTS_PATH),
    'class_names': ['fertile_spikelet', 'infertile_spikelet'],
    'num_classes': 2,
    'val_split': 0.18,

    # Training
    'image_size': 1920,
    'batch_size': 2,
    'epochs': 600,
    'patience': 180,
    'save_period': 50,

    # Optimizer
    'optimizer': 'AdamW',
    'initial_lr': 0.0005,
    'weight_decay': 1e-4,

    # Warmup
    'warmup_epochs': 5,
    'warmup_momentum': 0.5,
    'warmup_bias_lr': 0.05,

    # Class balance — per-class loss weighting
    'class_weights': [1.0, 6.6],
    'cls': 0.5,      # Default
    'copy_paste': 0.0,

    # Augmentation
    'hsv_h': 0.01,
    'hsv_s': 0.5,
    'hsv_v': 0.3,
    'scale': 0.3,
    'mosaic': 1.0,
    'mixup': 0.05,
    'close_mosaic': 30,
    'workers': 4,
    # AMP
    'amp': True,
}

# INFERENCE CONFIG
INFERENCE_CONFIG = {
    'spike_image_size': 2560,
    'spike_conf_threshold': 0.5,
    'spikelet_image_size': 1920,
    'spikelet_conf_threshold': 0.25,
    'spikelet_iou_threshold': 0.45,
    'spikelet_nms_threshold': 0.05,
    'augment': False,
    'agnostic_nms': False,
    'max_det': 300,
    'line_width': 2,
    'show_boxes': True,
    'show_labels': True,
    'show_conf': True,
}
# DEVICE CONFIGURATION
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

# DATA PREPARATION CONFIG
DATA_PREP_CONFIG = {
    'val_split': 0.18,
    'seed': 42,
    'balance_classes': False,
    'infertile_oversample': 1,
    'min_area': 50,
    'max_aspect_ratio': 5.0,
    'check_labels': True,
    'remove_invalid': True,
}
