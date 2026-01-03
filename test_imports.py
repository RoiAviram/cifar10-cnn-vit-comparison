# test_imports.py - Run this to find which import is hanging

print("Testing imports one by one...")
print()

print("1. Standard library...")
import argparse
import random
import shutil
import tempfile
from pathlib import Path
print("   OK")

print("2. NumPy...")
import numpy as np
print("   OK")

print("3. Matplotlib...")
import matplotlib.pyplot as plt
print("   OK")

print("4. PyTorch...")
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
print("   OK")

print("5. Torchvision...")
from torchvision import transforms
print("   OK")

print("6. tqdm...")
from tqdm import tqdm
print("   OK")

print("7. sklearn (this might be slow)...")
from sklearn.metrics import confusion_matrix, classification_report
print("   OK")

print("8. HuggingFace datasets...")
from datasets import load_dataset
print("   OK")

print()
print("=" * 40)
print("All imports successful!")
print("=" * 40)
