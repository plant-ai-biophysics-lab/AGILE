import sys
sys.setrecursionlimit(1000000)  # Set this to a higher value

import random
import torch
import math
import torchvision.datasets
import lightning as pl
import numpy as np

from torch.utils.data import Subset, DataLoader
from sklearn.cluster import KMeans
from typing import List
from sklearn.impute import SimpleImputer
from src.util import get_batchbald_batch
from tqdm import tqdm

class UncertaintySampling():
    """Active learning class for uncertainty sampling
    
    """
    used_samples = []
    
    def __init__(self, dataset: torchvision.datasets, verbose=False):
        self.dataset = dataset
        self.verbose = verbose
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.imputer = SimpleImputer(strategy='mean')
        
    def sample(
        self, 
        chunk: float, 
        method: str, 
        model: pl.LightningModule = None, 
        dm: pl.LightningDataModule = None, 
        batch_size: int = 64
    ):
        """Sample a chunk of the dataset using a specific method
        
        Args:
            chunk (float): a percent (fractional) of the dataset to sample
            method (str): the method to use for sampling
            model (pl.LightningModule): a PyTorch Lightning model to get probabilities
            batch_size (int): batch size for prediction
        """
        if method == 'random':
            if model is None:
                return self.random_sample(chunk = 0.1) # seed model
            else:
                return self.random_sample(chunk)
        elif method == 'entropy':
            if model is None:
                return self.random_sample(chunk = 0.1) # seed model
            else:
                return self.entropy_based(chunk, model, dm, batch_size)
        elif method == 'entropy_cluster':
            if model is None:
                return self.random_sample(chunk = 0.1) # seed model
            else:
                return self.entropy_cluster_based(chunk, model, dm, batch_size, 
                                                  num_clusters=20, max_iter=20)
        elif method == 'BatchBALD':
            if model is None:
                return self.random_sample(chunk = 0.1) # seed model
            else:
                return self.BatchBALD(chunk, model, dm, batch_size)
        else:
            raise ValueError("Invalid sampling method.")
        
    def random_sample(self, chunk: float):
        """Randomly sample a chunk of the dataset

        Args:
            chunk (float): a percent (fractional) of the dataset to sample

        Returns:
            torchvision.datasets.Subset: a subset of the full dataset
        """
        used_samples = self.get_used_samples()
        if chunk < 1.0:
            n_train = int(len(self.dataset) * chunk)
            all_indices = set(range(len(self.dataset)))
            available_indices = list(all_indices - set(used_samples))
            
            if len(available_indices) < n_train:
                raise ValueError("Not enough available indices to sample the required number of examples.")
            
            # create the new dataset with the selected samples (train set)
            samples = random.sample(available_indices, n_train)
            self.set_used_samples(used_samples + samples)
            train_dataset = Subset(self.dataset, used_samples + samples)
            
            # create the new dataset with the remaining samples (pool set)
            test_dataset = Subset(self.dataset, list(all_indices - set(used_samples+samples)))
        
            return train_dataset, test_dataset
        
    def entropy_based(
        self, chunk: float, 
        model: pl.LightningModule = None, 
        dm: pl.LightningDataModule = None, 
        batch_size: int = 64
    ):
        """ Returns uncertainty score of a probability distribution using entropy 
            
            Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                            
            Args:
                chunk (float): a percent (fractional) of the dataset to sample
                model (pl.LightningModule): a PyTorch Lightning model to get probabilities
                batch_size (int): batch size for prediction
                
            Reference: https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
        """
        # get available samples
        used_samples = self.get_used_samples()
        if chunk < 1.0:
            n_train = int(len(self.dataset) * chunk)
            all_indices = set(range(len(self.dataset)))
            available_indices = list(all_indices - set(used_samples))
            
            if len(available_indices) < n_train:
                raise ValueError("Not enough available indices to sample the required number of examples.")
            
        # prepare subset of samples for prediction
        subset = Subset(self.dataset, available_indices)
        if dm:
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=dm.collate_fn)
        else:
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        model.eval()
        
        entropies = []
        indices = []
        
        # get entropy of each sample
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                _, probs = model.predict_step(inputs, dropout=False)
                
                # calculate entropy
                if isinstance(probs, list): # check if probs is a list (e.g. for object detection)
                    for prob in probs:
                        raw_entropy = -torch.sum(prob * torch.log2(prob + 1e-5), dim=1) # calcualte the entropy of each detection of each image
                        normalized_entropy = raw_entropy / math.log2(prob.size(1)) # shape: (num detections in image)
                        normalized_entropy = torch.mean(normalized_entropy, dim=0) # average the entropy of all the detections in each image [1 number]
                        normalized_entropy = np.array([normalized_entropy.cpu()])
                else:
                    raw_entropy = -torch.sum(probs * torch.log2(probs + 1e-5), dim=1) # TODO: Calculate the entropy per box of each image.
                    normalized_entropy = raw_entropy / math.log2(probs.size(1))       # TODO: Average the entropy of all the boxes in each image.
                    normalized_entropy = np.array([normalized_entropy.cpu()])
                entropies.extend(normalized_entropy)
                indices.extend([available_indices[i] for i in range(len(inputs))])
        
        # sort indices by highest entropy
        sorted_indices = [x for _, x in sorted(zip(entropies, indices), reverse=True)]
        
        # select top samples based on entropy
        selected_indices = sorted_indices[:n_train]
        
        # update used samples
        self.set_used_samples(used_samples + selected_indices)
        
        # create the new dataset with the selected samples (train set)
        train_dataset = Subset(self.dataset, used_samples + selected_indices)
        
        # create the new dataset with the remaining samples (pool set)
        unselected_indices = list(all_indices - set(used_samples + selected_indices))
        test_dataset = Subset(self.dataset, unselected_indices)
        
        return train_dataset, test_dataset
    
    def entropy_cluster_based(
        self, 
        chunk: float, 
        model: pl.LightningModule = None, 
        dm: pl.LightningDataModule = None, 
        batch_size: int = 64,
        num_clusters: int = 20, 
        max_iter: int = 10
    ):
        
        # get available samples
        used_samples = self.get_used_samples()
        if chunk < 1.0:
            n_train = int(len(self.dataset) * chunk)
            all_indices = set(range(len(self.dataset)))
            available_indices = list(all_indices - set(used_samples))
            
            if len(available_indices) < n_train:
                raise ValueError("Not enough available indices to sample the required number of examples.")
        
        # prepare subset of samples for prediction
        subset = Subset(self.dataset, available_indices)
        if dm:
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=dm.collate_fn)
        else:
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        model.eval()
        
        # extract relevant features
        features = []
        entropies = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                logits, probs = model.predict_step(inputs, dropout=False)
                
                # calculate entropy
                if isinstance(probs, list): # check if probs is a list (e.g. for object detection)
                    for prob in probs:
                        raw_entropy = -torch.sum(prob * torch.log2(prob + 1e-5), dim=1) # calcualte the entropy of each detection of each image
                        normalized_entropy = raw_entropy / math.log2(prob.size(1)) # shape: (num detections in image)
                        normalized_entropy = torch.mean(normalized_entropy, dim=0) # average the entropy of all the detections in each image [1 number]
                        normalized_entropy = np.array([normalized_entropy.cpu()])
                else:
                    raw_entropy = -torch.sum(probs * torch.log2(probs + 1e-5), dim=1) # TODO: Calculate the entropy per box of each image.
                    normalized_entropy = raw_entropy / math.log2(probs.size(1))       # TODO: Average the entropy of all the boxes in each image.
                    normalized_entropy = np.array([normalized_entropy.cpu()])
                    
                # format logits if it is a list (e.g. for object detection)
                if isinstance(logits, list):
                    # average logits in each image
                    for i in range(len(logits)):
                        logits[i] = torch.mean(logits[i], dim=0)
                    # convert list to tensor
                    logits = torch.stack(logits)
                    
                features.append(logits.detach())
                entropies.extend(normalized_entropy)
            
        # flatten the list of features
        features = torch.cat(features).cpu().numpy()
        features = self.imputer.fit_transform(features)
        
        if features is not None and features.shape[1] > 0:
            # perform K-means clustering on the features
            kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=42)
            kmeans.fit(features)
            cluster_labels = kmeans.labels_
            
            # calculate average uncertainty for each cluster
            clusters = {i: [] for i in range(num_clusters)}
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(entropies[idx])
                
            # identify the cluster with the highest average normalized uncertainty
            highest_avg_uncertainty = -1
            most_uncertain_cluster = None
            for cluster, scores in clusters.items():
                if scores:
                    avg_uncertainty = sum(scores) / len(scores)
                    if avg_uncertainty > highest_avg_uncertainty:
                        highest_avg_uncertainty = avg_uncertainty
                        most_uncertain_cluster = cluster
                        
            # sample items from the most uncertain cluster
            indices = [i for i, label in enumerate(cluster_labels) if label == most_uncertain_cluster]
            selected_indices = random.sample(indices, min(len(indices), n_train))
            
            # create the new dataset with the selected samples (train set)
            self.set_used_samples(used_samples + selected_indices)
            train_dataset = Subset(self.dataset, used_samples + selected_indices)
                
            # create the new dataset with the remaining samples (pool set)
            unselected_indices = list(all_indices - set(used_samples + selected_indices))
            test_dataset = Subset(self.dataset, unselected_indices)
            
        else:
            # random sample
            print("NO FEATURES - RANDOM SAMPLING INSTEAD")
            train_dataset, test_dataset = self.random_sample(chunk)
        
        return train_dataset, test_dataset
    
    def BatchBALD(
        self,
        chunk: float,
        model: pl.LightningModule = None,
        dm: pl.LightningDataModule = None,
        batch_size: int = 64,
        num_samples: int = 2  # number of MC samples
    ):
        """Batch Bayesian Active Learning by Disagreement (BatchBALD) method for active learning.
        Selecting the most informative samples from a pool of data.
        
        Paper: https://arxiv.org/abs/1906.08158

        Args:
            chunk (float): Fraction of the dataset to use.
            model (pl.LightningModule, optional): Model to use for predictions. Defaults to None.
            dm (pl.LightningDataModule, optional): Data module. Defaults to None.
            batch_size (int, optional): Batch size for data loader. Defaults to 64.
            num_samples (int, optional): Number of Monte Carlo samples. Defaults to 10.
            acquisition_size (int, optional): Number of points to acquire in each batch. Defaults to 10.

        Raises:
            ValueError: If there are not enough available indices to sample the required number of examples.
        """
        
        # get available samples
        used_samples = self.get_used_samples()
        all_indices = set(range(len(self.dataset)))
        available_indices = list(all_indices - set(used_samples))
        
        if chunk < 1.0:
            n_train = int(len(self.dataset) * chunk)
            if len(available_indices) < n_train:
                raise ValueError("Not enough available indices to sample the required number of examples.")
        else:
            n_train = len(available_indices)  # When chunk is 1.0 or higher, use all available samples
        
        # get random subset from pool (computational reasons)
        p_samples = random.sample(available_indices, n_train*2)
        
        # prepare subset of samples for prediction
        subset = Subset(self.dataset, p_samples)
        if dm is not None:
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=dm.collate_fn)
        else:
            raise ValueError("Data module is required for BatchBALD.")
        model.eval()
        
        all_probs = []
        
        # get class probabilities for each sample
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                inputs, _ = batch
                inputs = inputs.to(self.device)
                
                try:
                    # multiple forward passes to get MC samples
                    batch_probs = []
                    for _ in range(num_samples):
                        _, probs = model.predict_step(inputs.clone(), dropout=True)  # length: batch_size / shape for each index: (num_detections, num_classes)
                        # log_probs = torch.log(probs) # TODO: dimensions of probs could be different
                        # replace probs with random probs if empty
                        for prob in probs:
                            if prob.nelement() == 0:
                                prob = torch.rand(1, model.hparams.num_classes - 1)
                        log_probs = [torch.log(prob).cpu() for prob in probs]
                        # batch_probs.append(log_probs.cpu())
                        batch_probs.append(log_probs) # batch_probs length: num_samples

                    # batch_probs = torch.stack(batch_probs, dim=1)  # shape: [batch_size, num_samples, num_classes]
                    all_probs.append(batch_probs) # all_probs length: len(dataloader) / length for each index: [num_samples]
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('CUDA out of memory. Attempting to free memory and continue.')
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                    
        # again check for empty probs
        for i, probs in enumerate(all_probs):
            for j, prob in enumerate(probs):
                for k, p in enumerate(prob):
                    if p.nelement() == 0:
                        all_probs[i][j][k] = torch.log(torch.rand(1, model.hparams.num_classes - 1)).cpu()
                        
        # Compute BatchBALD batch
        candidate_batch = get_batchbald_batch(
            log_probs_N_K_C = all_probs, 
            n_train = n_train, 
            num_samples = num_samples, 
            num_classes = model.hparams.num_classes, 
            batch_size=batch_size,
            N = len(p_samples)
        )

        selected_indices = [p_samples[i] for i in candidate_batch.indices]

        # update used samples
        self.set_used_samples(used_samples + selected_indices)

        # create the new dataset with the selected samples (train set)
        train_dataset = Subset(self.dataset, used_samples + selected_indices)

        # create the new dataset with the remaining samples (pool set)
        unselected_indices = list(all_indices - set(used_samples + selected_indices))
        test_dataset = Subset(self.dataset, unselected_indices)
        
        # return model to original state
        model._return_model()
        
        return train_dataset, test_dataset
        
    @classmethod
    def set_used_samples(cls, used_samples):
        cls.used_samples = used_samples
        
    @classmethod
    def get_used_samples(cls):
        return cls.used_samples
