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
            
        # get original source image size
        source_image = Image.open(os.path.join(self.source_images_path, self.source_image_files[0])).convert("RGB")
        self.source_image_size = source_image.size
        
        # get original target image size
        target_image = Image.open(os.path.join(self.target_images_path, self.target_image_files[0])).convert("RGB")
        self.target_image_size = target_image.size
    
    def balance_dataset_lengths(self):
        if len(self.source_image_files) < len(self.target_image_files):
            deficit = len(self.target_image_files) - len(self.source_image_files)
            self.source_image_files += random.choices(self.source_image_files, k=deficit)
        elif len(self.target_image_files) < len(self.source_image_files):
            deficit = len(self.source_image_files) - len(self.target_image_files)
            self.target_image_files += random.choices(self.target_image_files, k=deficit)
   
    @staticmethod
    def gaussian_map(image, yolo_file):
        # Create an empty attention map
        attention_map = np.zeros(image.shape[:2])
        
        # Get bounding box coordinates
        with open(yolo_file, 'r') as f:
            label = f.readlines()
        
        for line in label:
            _, x_center, y_center, width, height = map(float, line.split())
            
            # Convert to image coordinates
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            
            # Make sure the bounding box is within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1] - 1, x_max)
            y_max = min(image.shape[0] - 1, y_max)
            
            # Calculate the bounding box size
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Set sigma to ensure the Gaussian falls to near-zero at the edges
            sigma_x = box_width / 4.0
            sigma_y = box_height / 4.0
            
            # Create a grid of coordinates within the bounding box
            x_coords = np.linspace(x_min, x_max, box_width)
            y_coords = np.linspace(y_min, y_max, box_height)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            
            # Calculate distances from the center of the bounding box
            x_center_image = x_center * image.shape[1]
            y_center_image = y_center * image.shape[0]
            dist_squared_x = ((x_grid - x_center_image) / sigma_x) ** 2
            dist_squared_y = ((y_grid - y_center_image) / sigma_y) ** 2
            dist_squared = dist_squared_x + dist_squared_y
            
            # Apply Gaussian to create a peak of 1 at the center, falling to zero at the edges
            gaussian_map = np.exp(-dist_squared / 2)
            gaussian_map = np.clip(gaussian_map, 0, 1)  # Ensure values are within [0, 1]
            
            # Place the Gaussian in the attention map
            attention_map[y_min:y_max, x_min:x_max] = np.maximum(attention_map[y_min:y_max, x_min:x_max], gaussian_map)
        
        # Normalize the entire attention map (if needed)
        if np.max(attention_map) > 0:
            attention_map /= np.max(attention_map)
        
        # Convert to PIL image
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