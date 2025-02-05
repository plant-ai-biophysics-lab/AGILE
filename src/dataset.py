import os
import random
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from PIL import Image

class ControlNetDataset(Dataset):
    def __init__(
        self, source_images_path, target_images_path, prompt, transform=None, optimizing=False, 
        spread_factor=4.0, betas=None, generated=None, img_size=512, use_transforms=False, apply_bbox_mask=False
    ):
        self.source_images_path = source_images_path
        self.target_images_path = target_images_path
        self.prompt = prompt
        self.transform = transform
        self.optimizing = optimizing
        self.spread_factor = spread_factor
        self.betas=betas
        self.generated = generated
        self.img_size = img_size
        self.use_transforms = use_transforms
        self.apply_bbox_mask = apply_bbox_mask
        
        # Load image file paths
        if self.generated is not None:
            self.source_image_files = [f for f in os.listdir(self.generated) if f.endswith(('jpg', 'jpeg', 'png'))]
            self.target_image_files = [f for f in os.listdir(self.generated) if f.endswith(('jpg', 'jpeg', 'png'))]
        else:
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
                'control': None, # placeholder for control
                'betas': self.betas
            })
            
        # get original source image size
        source_image = Image.open(os.path.join(self.source_images_path, self.source_image_files[0])).convert("RGB")
        self.source_image_size = source_image.size
        target_image = Image.open(os.path.join(self.target_images_path, self.target_image_files[0])).convert("RGB")
        self.target_images_size = target_image.size
        
        self.geometric_augmentation = A.Compose([
            A.RandomCrop(height=int(self.img_size * 0.8), width=int(self.img_size * 0.8), p=1.0),
            A.Resize(self.img_size, self.img_size)  
        ], additional_targets={"image0": "image"})
        self.color_augmentation = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.2), contrast_limit=(-0.4, 0.2), p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5)
        ])
        
    @staticmethod
    def read_yolo_labels(label_path, img_width, img_height):
        """
        Reads YOLO labels and converts them to absolute pixel bounding boxes.
        
        Args:
            label_path (str): Path to the YOLO label file (.txt).
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            list of tuples: List of bounding boxes in absolute pixel format (x_min, y_min, x_max, y_max).
        """
        bboxes = []
        with open(label_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts)
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                x_max = int((x_center + width / 2) * img_width)
                y_max = int((y_center + height / 2) * img_height)
                bboxes.append((x_min, y_min, x_max, y_max))
        return bboxes
    
    def mask_bbox(self, image, yolo_file):
        """
        Applies a white mask over the bounding box areas specified in the YOLO label file.

        Args:
            image (np.array): The input image.
            yolo_file (str): Path to the YOLO label file.

        Returns:
            np.array: The masked image.
        """
        img_array = np.array(image).astype(np.uint8)
        img_height, img_width = img_array.shape[:2]

        bboxes = self.read_yolo_labels(yolo_file, img_width, img_height)

        for x_min, y_min, x_max, y_max in bboxes:
            img_array[y_min:y_max, x_min:x_max] = 255  # Apply white mask
            
        # TEMP: Save masked image for debugging
        Image.fromarray(img_array).save("masked_output.jpg")
        print("Masked image saved as 'masked_output.jpg'")

        return img_array
        
    def random_mask(self, image):
        """
        Randomly masks out 3 regions of the image with rectangular boxes.

        Args:
            image (np.array): The input image (expected in [-1,1] normalized format).
            label_path (str): Path to the YOLO label file (not used anymore).

        Returns:
            np.array: The masked image in the same format as the input.
        """
        # # De-normalize image from [-1,1] back to [0,255]
        # img_array = ((image + 1.0) * 127.5).astype(np.uint8)
        img_array = np.array(image).astype(np.uint8)

        img_height, img_width = img_array.shape[:2]

        # Define the number of random masks
        num_masks = 3
        max_mask_size = (img_width // 4, img_height // 4)  # Maximum size of each mask

        for _ in range(num_masks):
            # Randomly select a position for the mask
            x_min = random.randint(0, img_width - max_mask_size[0])
            y_min = random.randint(0, img_height - max_mask_size[1])

            # Randomly determine the box size (ensuring it doesn't exceed max_mask_size)
            box_width = random.randint(max_mask_size[0] // 2, max_mask_size[0])
            box_height = random.randint(max_mask_size[1] // 2, max_mask_size[1])

            x_max = min(img_width, x_min + box_width)
            y_max = min(img_height, y_min + box_height)

            # Apply hard mask (white) to the randomly chosen bounding box
            img_array[y_min:y_max, x_min:x_max] = 255  # Set to white (255)

        # # TEMP: Save masked image for debugging
        # Image.fromarray(img_array).save("masked_output.jpg")
        # print("Masked image saved as 'masked_output.jpg'")

        return img_array
    
    def balance_dataset_lengths(self):
        if len(self.source_image_files) < len(self.target_image_files):
            deficit = len(self.target_image_files) - len(self.source_image_files)
            self.source_image_files += random.choices(self.source_image_files, k=deficit)
        elif len(self.target_image_files) < len(self.source_image_files):
            deficit = len(self.source_image_files) - len(self.target_image_files)
            self.target_image_files += random.choices(self.target_image_files, k=deficit)
   
    def gaussian_map(self, image, yolo_file):
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
            
            # Ensure bounding box stays within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1] - 1, x_max)
            y_max = min(image.shape[0] - 1, y_max)
            
            # Calculate the bounding box size
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Set sigma to ensure the Gaussian falls to near-zero at the edges
            sigma_x = box_width / self.spread_factor
            sigma_y = box_height / self.spread_factor
            
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
        
        # Add random noise to the background (outside bounding boxes)
        background_mask = attention_map == 0  # Mask for the background (where attention_map is 0)
        random_noise = np.random.random(image.shape[:2])  # Generate random noise in [0, 1]
        attention_map[background_mask] = random_noise[background_mask]  # Assign noise to background
        
        # Separate foreground and background maps
        foreground_map = np.where(background_mask, 0, attention_map)  # Foreground (Gaussian regions)
        background_map = np.where(background_mask, attention_map, 0)  # Background (noise regions)
        
        # Convert to PIL images
        foreground_map = Image.fromarray((foreground_map * 255).astype(np.uint8))
        background_map = Image.fromarray((background_map * 255).astype(np.uint8))
        
        return {'object': foreground_map, 'background': background_map}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        source_image_file = item['source']
        target_image_file = item['target']
        prompt = item['prompt']
        betas = item['betas']

        if self.generated is not None:
            source_image_path = os.path.join(self.generated, source_image_file)
            target_image_path = os.path.join(self.generated, target_image_file)
            original_source_path = os.path.join(self.source_images_path, source_image_file)
            yolo_file = original_source_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        else:
            source_image_path = os.path.join(self.source_images_path, source_image_file)
            target_image_path = os.path.join(self.target_images_path, target_image_file)
            yolo_file = source_image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        
        # Load images
        source_image = Image.open(source_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
            
        # Get attention map
        attn_map = self.gaussian_map(np.array(source_image), yolo_file)
        
        # Apply transformations if any
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
            attn_map['object'] = self.transform(attn_map['object'])
            attn_map['background'] = self.transform(attn_map['background'])
        
        # Normalize attention map between 0 and 1
        attn_map['object'] = np.array(attn_map['object']).astype(np.float32) / 255.0
        attn_map['background'] = np.array(attn_map['background']).astype(np.float32) / 255.0
        
        if self.optimizing:
            target_image = source_image
            
            # Normalize images
            source_image = np.array(source_image).astype(np.float32) / 127.5 - 1.0
            target_image = np.array(target_image).astype(np.float32) / 127.5 - 1.0
            
        else:
            source_image = target_image
            
            if self.use_transforms:
                source_image_np = np.array(source_image)
                target_image_np = np.array(target_image)

                transformed = self.geometric_augmentation(image=source_image_np, image0=target_image_np)
                source_image = transformed["image"]
                target_image = transformed["image0"]
                source_image = self.color_augmentation(image=source_image)["image"]

                # Apply Masking AFTER Augmentation
                source_image = self.random_mask(source_image)
                
            ## TODO: ADD MASK BBOX HERE IF TRUE
            if self.apply_bbox_mask:
                source_image = self.mask_bbox(source_image, yolo_file)
                
            # Normalize images
            source_image = np.array(source_image).astype(np.float32) / 127.5 - 1.0
            target_image = np.array(target_image).astype(np.float32) / 127.5 - 1.0
            
        return dict(
            jpg=target_image,
            txt=prompt,
            hint=source_image,
            source_path=source_image_path,
            target_path=target_image_path,
            attn_map=attn_map,
            betas=betas
        )