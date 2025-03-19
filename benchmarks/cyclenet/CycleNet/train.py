import os
import argparse
import numpy as np

from torch.utils.data import Dataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cycleNet.logger import ImageLogger
from cycleNet.model import create_model, load_state_dict
from PIL import Image
from pathlib import Path
from torchvision import transforms

# Configs
batch_size_per_gpu = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = False
only_mid_control = False

class PermuteTransform:
    def __call__(self, x):
        # Permute dimensions from [512, 512, 3] to [512, 3, 512]
        # Only permute if 3 channels are present
        if len(np.array(x).shape) == 2:
            return x
        else:
            return np.transpose(x, (0, 1, 2))

class ControlNetDataset(Dataset):
    def __init__(
        self, source_prompt, target_images_path, prompt, transform=None, img_size=512,
        device="cuda"
    ):
        self.source_prompt = source_prompt
        self.target_images_path = target_images_path
        self.prompt = prompt
        self.transform = transform
        self.img_size = img_size
        self.device = device
        
        # Load image file paths
        self.target_image_files = [f for f in os.listdir(target_images_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        
        # Create dataset structure
        self.data = []
        for target_file in self.target_image_files:
            self.data.append({
                'source': self.source_prompt,
                'target': target_file,
                'prompt': self.prompt,
            })
            
        # get original source image size
        target_image = Image.open(os.path.join(self.target_images_path, self.target_image_files[0])).convert("RGB")
        self.target_images_size = target_image.size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        source_prompt = item['source']
        target_image_file = item['target']
        prompt = item['prompt']

        target_image_path = os.path.join(self.target_images_path, target_image_file)
        
        # Load image
        target_image = Image.open(target_image_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            target_image = self.transform(target_image)
        target_image = np.array(target_image).astype(np.float32) / 127.5 - 1.0
            
        return dict(
            jpg=target_image,
            txt=prompt,
            source=source_prompt,
        )

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", type=Path, required=True,
                    help="Path to model configuration file.")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to model checkpoint.")
    ap.add_argument("--image_size", type=int, default=512,
                    help="Size of images.")
    ap.add_argument("--source_prompt", type=str, required=True,
                    help="Source prompt text.")
    ap.add_argument("--target_images_path", type=Path, required=True,
                    help="Path to target images.")
    ap.add_argument("--prompt", type=str, required=True,
                    help="Prompt to use for image generation.")
    ap.add_argument("--logs_dir", type=Path, default="logs",
                    help="Directory to save logs.")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Number of epochs.")
    # New argument for specifying the number of GPUs
    ap.add_argument("--gpus", type=int, default=1,
                    help="Number of GPUs to use. Set >1 for multi-GPU training.")
    args = ap.parse_args()

    # Load model on CPU first; Lightning will move it to GPUs.
    model = create_model(args.model_config).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Define transformation and dataset
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        PermuteTransform()
    ])
    dataset = ControlNetDataset(
        source_prompt=args.source_prompt,
        target_images_path=args.target_images_path,
        prompt=args.prompt,
        transform=transform,
        img_size=args.image_size,
    )
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size_per_gpu, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq, every_n_train_steps=logger_freq)
    logger.train_dataloader = dataloader

    # Set up the Trainer.
    # When using more than one GPU, use the 'ddp' strategy.
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else None,
        precision=32,
        callbacks=[logger],
        default_root_dir=args.logs_dir
    )
    trainer.fit(model, dataloader)
