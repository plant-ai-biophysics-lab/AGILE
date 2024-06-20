import wandb
import os
import cv2
import torch
import numpy as np
import torchvision.datasets

import torchvision.transforms as transforms
import lightning as pl
import albumentations as A
import xml.etree.ElementTree as ET

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VOCDetection
from torch.utils.data import random_split

### MNIST DATAMODULE ###
class MNISTDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        train_dataset: torchvision.datasets, 
        test_dataset: torchvision.datasets, 
        root_dir: str, 
        batch_size: int = 32
    ) -> pl.LightningDataModule:
        """Data Module for handling MNIST dataset.
        User can download the dataset by using 

        Args:
            train_dataset (torchvision.datasets): Training dataset.
            test_dataset (torchvision.datasets): Testing dataset.
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
    
### PASCAL VOC DATASET ###
class PASCALDataset(Dataset):
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train',
        year: str = '2012',
        transform = None
    ) -> Dataset:
        
        super().__init__()
        self.split = split
        self.year = year
        self.transform = transform
        self.num_classes = 20
        
        # directory of images and labels
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'VOCdevkit', 'VOC' + year, 'JPEGImages')
        self.labels_dir = os.path.join(root_dir, 'VOCdevkit', 'VOC' + year, 'Annotations')
        
        # splits directory
        self.splits_dir = os.path.join(self.root_dir, 'VOCdevkit', 'VOC' + year, 'ImageSets', 'Main')
        
        # load splits
        self.img_ids = []
        self.images = []
        self.labels = []
        self.load_data()
        print('Loaded {} images and {} labels'.format(len(self.images), len(self.labels)))
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        
        # load image
        image = cv2.imread(self.images[idx], cv2.COLOR_BGR2RGB)
        
        # load xml label as dictionary
        label_path = self.labels[idx]
        boxes, labels = self.parse_voc_xml(label_path)
        
        # add transform
        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = torch.tensor(augmented['bboxes'], dtype=torch.float32)
            labels = torch.tensor(augmented['labels'], dtype=torch.long)
        
        return image, {'boxes': boxes, 'labels': labels}
    
    def load_data(self):
        
        with open(os.path.join(self.splits_dir, self.split + '.txt')) as f:
            for line in f:
                self.img_ids.append(line.strip())
                
        for img_id in self.img_ids:
            img_file = os.path.join(self.img_dir, img_id + '.jpg')
            ann_file = os.path.join(self.labels_dir, img_id + '.xml')
            self.images.append(img_file)
            self.labels.append(ann_file)
            
        assert len(self.images) == len(self.labels)
        
    def parse_voc_xml(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            box = [
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text)
            ]
            boxes.append(box)
            labels.append(label)  # Labels are still strings here

        # Convert labels to integers
        label_map = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                     'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
                     'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
                     'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
        labels = [label_map[label] for label in labels]
        
        return boxes, labels
    
### PASCAL VOC DATAMODULE ###
class PASCALDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        root_dir: str, 
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 32, 
        num_workers: int = 4, 
        image_size: int = 512
    ) -> pl.LightningDataModule:
        """Data Module for handling PASCAL VOC 2012 dataset.

        Args:
            root_dir (str): Root directory of the PASCAL VOC dataset.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            train_dataset (Dataset): Training dataset.
            test_dataset (Dataset): Testing dataset.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            num_workers (int, optional): Number of workers to use for data loading. Defaults to 4.
            image_size (int, optional): Size of the image to use. Defaults to 512.
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset", "test_dataset"])
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.train_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        self.test_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def prepare_data(self):
        # Download the Pascal VOC dataset
        VOCDetection(self.root_dir, year='2012', image_set='train', download=False)
        VOCDetection(self.root_dir, year='2012', image_set='val', download=False)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print("Setting up training datasets")
            if self.train_dataset is None:
                self.train_dataset = PASCALDataset(self.root_dir, split='train', transform=self.train_transform)
            wandb.log({"train_size": len(self.train_dataset)})
        
        if stage == 'test' or stage is None:
            if self.test_dataset is None:
                self.test_dataset = PASCALDataset(self.root_dir, split='val', transform=self.test_transform)
            wandb.log({"test_size": len(self.test_dataset)})

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        boxes = [target['boxes'] for target in targets]
        labels = [target['labels'] for target in targets]
        return images, {'boxes': boxes, 'labels': labels}
