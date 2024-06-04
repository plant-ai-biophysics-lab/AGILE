import wandb
import torchvision.datasets

import torchvision.transforms as transforms
import lightning as pl

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_dataset: torchvision.datasets, 
        test_dataset: torchvision.datasets, 
        root_dir: str, 
        batch_size: int = 32
    ):
        """Data Module for handling MNIST dataset.
        User can download the dataset by using 

        Args:
            root_dir (str): Root directory of the MNIST dataset.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self):
        MNIST(self.root_dir, train=True, download=False)
        MNIST(self.root_dir, train=False, download=False)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # training dataloader
            if self.train_dataset is not None:
                self.train_dataset.dataset.transform = self.transform
                self.train_set = self.train_dataset
            else:
                self.train_set = MNIST(self.root_dir, train=True, transform=self.transform)
            wandb.log({"train_size": len(self.train_set)})
            
            # val dataloader
            self.val_set = MNIST(self.root_dir, train=False, transform=self.transform)
            wandb.log({"val_size": len(self.val_set)})
                
        if stage == 'test' or stage is None:
            if self.test_dataset is not None:
                self.test_dataset.dataset.transform = self.transform
                self.test_set = self.test_dataset
            else:
                self.test_set = MNIST(self.root_dir, train=False, transform=self.transform)
            wandb.log({"test_size": len(self.test_set)})

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)