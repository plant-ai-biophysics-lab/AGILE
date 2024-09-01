import os
import torch
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from src.modules import MaskImage

class ControlNetDataset(Dataset):
    def __init__(self, source_images_path, target_images_path, transform=None):
        self.source_images_path = source_images_path
        self.target_images_path = target_images_path
        self.transform = transform
        self.mask_image = MaskImage()
        
        # Load image file paths
        self.source_image_files = [f for f in os.listdir(source_images_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        self.target_image_files = [f for f in os.listdir(target_images_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        
        # Balance the dataset lengths
        self.balance_dataset_lengths()
        
        # Create dataset structure
        self.data = []
        for source_file, target_file in zip(self.source_image_files, self.target_image_files):
            self.data.append({
                'source': source_file,
                'target': target_file,
                'prompt': 'grape',
                'control': None # placeholder for control
            })
    
    def balance_dataset_lengths(self):
        if len(self.source_image_files) < len(self.target_image_files):
            deficit = len(self.target_image_files) - len(self.source_image_files)
            self.source_image_files += random.choices(self.source_image_files, k=deficit)
        elif len(self.target_image_files) < len(self.source_image_files):
            deficit = len(self.source_image_files) - len(self.target_image_files)
            self.target_image_files += random.choices(self.target_image_files, k=deficit)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        source_image_file = item['source']
        target_image_file = item['target']
        prompt = item['prompt']

        source_image_path = os.path.join(self.source_images_path, source_image_file)
        target_image_path = os.path.join(self.target_images_path, target_image_file)
        
        # Load images
        source_image = Image.open(source_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
            
        # Get segmentation mask
        yolo_file = source_image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        if os.path.exists(yolo_file):
            mask = self.mask_image(np.array(source_image), yolo_file)
        
        # Apply transformations if any
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
            mask = self.transform(mask)
            
        # # Normalize images
        mask = np.array(mask).astype(np.float32) / 127.5 - 1.0
        source_image = np.array(source_image).astype(np.float32) / 127.5 - 1.0
        target_image = np.array(target_image).astype(np.float32) / 127.5 - 1.0

        # return dict(jpg=target_image, txt=prompt, hint=source_image)
        # return dict(jpg=source_image, txt=prompt, control=source_image, hint=mask)
        return dict(jpg=source_image, txt=prompt, hint=source_image)