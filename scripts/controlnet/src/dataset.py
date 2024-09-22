import os
import torch
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class ControlNetDataset(Dataset):
    def __init__(self, source_images_path, target_images_path, prompt, transform=None, optimizing=False):
        self.source_images_path = source_images_path
        self.target_images_path = target_images_path
        self.prompt = prompt
        self.transform = transform
        self.optimizing = optimizing
        
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
                'prompt': self.prompt,
                'control': None # placeholder for control
            })
    
    def balance_dataset_lengths(self):
        if len(self.source_image_files) < len(self.target_image_files):
            deficit = len(self.target_image_files) - len(self.source_image_files)
            self.source_image_files += random.choices(self.source_image_files, k=deficit)
        elif len(self.target_image_files) < len(self.source_image_files):
            deficit = len(self.source_image_files) - len(self.target_image_files)
            self.target_image_files += random.choices(self.target_image_files, k=deficit)
    
    @staticmethod
    def gaussian_map(image, yolo_file, sigma=20.0):
        
        # Create an empty attention map
        attention_map = np.zeros(image.shape[:2])
        
        # Create a grid of coordinates
        x_coords = np.arange(image.shape[1])
        y_coords = np.arange(image.shape[0])
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Get center points
        with open(yolo_file, 'r') as f:
            label = f.readlines()
        center_points = []
        for line in label:
            _, x_center, y_center, _, _ = map(float, line.split())
            center_points.append((x_center * image.shape[1], y_center * image.shape[0]))
            
        # Generate attention map
        for x, y in center_points:
            dist_squared = (x_grid - x) ** 2 + (y_grid - y) ** 2
            attention_map += np.exp(-dist_squared / (2 * sigma ** 2))
            
        attention_map /= np.max(attention_map)
        
        # Change to PIL image
        attention_map = Image.fromarray((attention_map * 255).astype(np.uint8))
            
        return attention_map
    
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
            
        # Get attention map
        yolo_file = source_image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        attn_map = self.gaussian_map(np.array(source_image), yolo_file)
        
        # Apply transformations if any
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
            attn_map = self.transform(attn_map)
            
        # Normalize images
        source_image = np.array(source_image).astype(np.float32) / 127.5 - 1.0
        target_image = np.array(target_image).astype(np.float32) / 127.5 - 1.0
        
        # Normalize attention map between 0 and 1
        attn_map = np.array(attn_map).astype(np.float32) / 255.0
        
        if self.optimizing:
            target_image = source_image

        return dict(
            jpg=target_image,
            txt=prompt,
            hint=source_image,
            source_path=source_image_path,
            target_path=target_image_path,
            attn_map=attn_map,
        )