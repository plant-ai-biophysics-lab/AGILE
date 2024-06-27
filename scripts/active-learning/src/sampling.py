import random
import torch
import math
import torchvision.datasets

import lightning as pl

from torch.utils.data import Subset, DataLoader
from sklearn.cluster import KMeans
from typing import List
from sklearn.impute import SimpleImputer
from torch.distributions import Categorical

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
            return self.random_sample(chunk)
        elif method == 'entropy':
            if model is None:
                return self.random_sample(chunk)
            else:
                return self.entropy_based(chunk, model, dm, batch_size)
        elif method == 'entropy_cluster':
            if model is None:
                return self.random_sample(chunk)
            else:
                return self.entropy_cluster_based(chunk, model, dm, batch_size, 
                                                  num_clusters=20, max_iter=20)
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
                _, probs = model.predict_step(inputs)
                
                # calculate entropy
                raw_entropy = -torch.sum(probs * torch.log2(probs + 1e-5), dim=1)
                normalized_entropy = raw_entropy / math.log2(probs.size(1))
                entropies.extend(normalized_entropy.cpu().numpy())
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
                logits, probs = model.predict_step(inputs)
                
                # calculate entropy
                raw_entropy = -torch.sum(probs * torch.log2(probs + 1e-5), dim=1)
                normalized_entropy = raw_entropy / math.log2(probs.size(1))
                features.append(logits.detach())
                entropies.extend(normalized_entropy.cpu().numpy())
            
        # flatten the list of features
        features = torch.cat(features).cpu().numpy()
        features = self.imputer.fit_transform(features)
        
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
        
        return train_dataset, test_dataset
    
    def BatchBALD(
        self,
        chunk: float,
        model: pl.LightningModule = None,
        dm: pl.LightningDataModule = None,
        batch_size: int = 64,
        num_samples: int = 100  # number of MC samples
    ):
        """Bayesian Active Learning by Disagreement (BALD) is a mutual-information-based method for active learning.
        Selecting the most informative samples from a pool of data.

        Args:
            chunk (float): _description_
            model (pl.LightningModule, optional): _description_. Defaults to None.
            dm (pl.LightningDataModule, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 64.
            num_samples (int, optional): _description_. Defaults to 100#numberofMCsamples.

        Raises:
            ValueError: _description_
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
        p_samples = random.sample(available_indices, n_train)
        
        # prepare subset of samples for prediction
        subset = Subset(self.dataset, p_samples)
        if dm:
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=dm.collate_fn)
        else:
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        model.eval()
        
        all_entropies = []
        all_log_probs = []
        
        # get class probabilities for each sample
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                
                # Multiple forward passes to get MC samples
                batch_log_probs = []
                for _ in range(num_samples):
                    logits, probs = self.predict_step(inputs, dropout=True)  # enable dropout during inference
                    batch_log_probs.append(probs.log())
                
                batch_log_probs = torch.stack(batch_log_probs, dim=1)  # Shape: [batch_size, num_samples, num_classes]
                all_log_probs.append(batch_log_probs.cpu())
                
                # calculate entropy for each detection
                raw_entropy = -torch.sum(probs * torch.log2(probs + 1e-5), dim=1)
                normalized_entropy = raw_entropy / math.log2(probs.size(1))
                all_entropies.append(normalized_entropy.cpu().numpy())
        
        all_log_probs = torch.cat(all_log_probs, dim=0)  # shape: [dataset_size, num_samples, num_classes]
        mean_log_probs = all_log_probs.mean(dim=1)  # shape: [dataset_size, num_classes]
        
        # calculate predictive entropy
        predictive_entropy = Categorical(logits=mean_log_probs).entropy()
        
        # calculate expected entropy
        expected_entropy = Categorical(logits=all_log_probs).entropy().mean(dim=1)
        
        # calculate mutual information
        mutual_information = predictive_entropy - expected_entropy
        
        # select top samples based on mutual information
        n_train = min(n_train, len(mutual_information))  # ensure n_train does not exceed the number of available samples
        _, top_indices = torch.topk(mutual_information, n_train)
        
        selected_indices = [p_samples[i] for i in top_indices]
        
        # update used samples
        self.set_used_samples(used_samples + selected_indices)
        
        # create the new dataset with the selected samples (train set)
        train_dataset = Subset(self.dataset, used_samples + selected_indices)
        
        # create the new dataset with the remaining samples (pool set)
        unselected_indices = list(all_indices - set(used_samples + selected_indices))
        test_dataset = Subset(self.dataset, unselected_indices)
        
        return train_dataset, test_dataset
        
    @classmethod
    def set_used_samples(cls, used_samples):
        cls.used_samples = used_samples
        
    @classmethod
    def get_used_samples(cls):
        return cls.used_samples