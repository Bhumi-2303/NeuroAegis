"""
Data pipeline for the NeuroAegis attention pooling architecture.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional

class FeatureNormalizer:
    """
    Computes per-feature mean and standard deviation for Z-score normalization.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, features_array: np.ndarray) -> None:
        """
        Compute mean and std per feature.
        
        Args:
            features_array (np.ndarray): Shape [N_samples, N_features].
        """
        self.mean = np.mean(features_array, axis=0)
        self.std = np.std(features_array, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1e-8
        
    def transform(self, features_array: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization.
        
        Args:
            features_array (np.ndarray): Shape [N_samples, N_features].
            
        Returns:
            np.ndarray: Normalized features.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted.")
        return (features_array - self.mean) / self.std
        
    def fit_transform(self, features_array: np.ndarray) -> np.ndarray:
        """
        Fit and apply Z-score normalization.
        
        Args:
            features_array (np.ndarray): Shape [N_samples, N_features].
            
        Returns:
            np.ndarray: Normalized features.
        """
        self.fit(features_array)
        return self.transform(features_array)

class MultiChannelEEGDataset(Dataset):
    """
    Dataset for loading multi-channel EEG features from parquet files.
    """
    def __init__(
        self,
        parquet_path: str,
        channel_prefix_pattern: str = r'^Ch\d+_',
        augment: bool = False,
        min_channel_keep_ratio: float = 0.3,
        normalize: bool = False,
        selected_features: Optional[List[str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            parquet_path (str): Path to parquet file.
            channel_prefix_pattern (str): Regex to detect channel columns.
            augment (bool): Whether to subsample channels.
            min_channel_keep_ratio (float): Min fraction of channels to keep.
            normalize (bool): Whether to apply feature normalization.
            selected_features (Optional[List[str]]): List of feature names to use.
        """
        self.df = pd.read_parquet(parquet_path)
        self.augment = augment
        self.min_channel_keep_ratio = min_channel_keep_ratio
        
        # Identify channel columns
        all_cols = self.df.columns.tolist()
        channel_cols = [c for c in all_cols if re.match(channel_prefix_pattern, c)]
        
        # Group by channel
        channel_prefixes = set()
        for c in channel_cols:
            match = re.match(r'^(Ch\d+)_', c)
            if match:
                channel_prefixes.add(match.group(1))
        
        # Sort to ensure consistent order
        self.channel_prefixes = sorted(list(channel_prefixes), key=lambda x: int(x[2:]))
        
        self.n_channels = len(self.channel_prefixes)
        
        # Get feature names from the first channel
        first_ch = self.channel_prefixes[0]
        extracted_feature_names = sorted([c[len(first_ch)+1:] for c in channel_cols if c.startswith(f"{first_ch}_")])
        
        if selected_features is not None:
            self.feature_names = selected_features
        else:
            self.feature_names = extracted_feature_names
            
        self.n_features = len(self.feature_names)
        
        # Prepare data array [N_samples, n_channels, n_features]
        N_samples = len(self.df)
        self.data_array = np.zeros((N_samples, self.n_channels, self.n_features), dtype=np.float32)
        
        for i, ch in enumerate(self.channel_prefixes):
            cols = [f"{ch}_{f}" for f in self.feature_names]
            self.data_array[:, i, :] = self.df[cols].values
            
        if normalize:
            # Flatten to [N_samples * n_channels, n_features] to fit normalizer
            flat_data = self.data_array.reshape(-1, self.n_features)
            self.normalizer = FeatureNormalizer()
            normalized_flat = self.normalizer.fit_transform(flat_data)
            self.data_array = normalized_flat.reshape(N_samples, self.n_channels, self.n_features)
            
        # Metadata
        self.labels = self.df['target'].values
        self.patient_ids = self.df['patient_id'].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dict containing features, label, patient_id, mask, n_channels.
        """
        features = self.data_array[idx] # [n_channels, 57]
        label = int(self.labels[idx])
        patient_id = str(self.patient_ids[idx])
        
        n_ch = self.n_channels
        
        if self.augment:
            # Randomly select subset of channels
            keep_prob = np.random.uniform(self.min_channel_keep_ratio, 1.0)
            n_keep = max(1, int(n_ch * keep_prob))
            
            # Select random indices
            keep_indices = np.random.choice(n_ch, size=n_keep, replace=False)
            keep_indices.sort()
            
            features = features[keep_indices]
            n_ch = n_keep
            
        mask = torch.ones(n_ch, dtype=torch.bool)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': label,
            'patient_id': patient_id,
            'mask': mask,
            'n_channels': n_ch
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to pad channels in a batch to max channels in the batch.
    
    Args:
        batch (List[Dict[str, Any]]): List of samples.
        
    Returns:
        Dict: Collated batch.
    """
    n_features = batch[0]['features'].shape[1]
    
    # Find max channels in this batch
    max_channels = max(item['n_channels'] for item in batch)
    
    batch_size = len(batch)
    
    padded_features = torch.zeros(batch_size, max_channels, n_features, dtype=torch.float32)
    padded_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    patient_ids = []
    n_channels = []
    
    for i, item in enumerate(batch):
        ch = item['n_channels']
        padded_features[i, :ch, :] = item['features']
        padded_mask[i, :ch] = item['mask']
        labels[i] = item['label']
        patient_ids.append(item['patient_id'])
        n_channels.append(ch)
        
    return {
        'features': padded_features,
        'labels': labels.unsqueeze(-1), # [B, 1]
        'mask': padded_mask,
        'patient_ids': patient_ids,
        'n_channels': n_channels
    }

def get_patient_splits(dataset: MultiChannelEEGDataset, n_folds: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns indices for Leave-One-Patient-Out Cross-Validation.
    
    Args:
        dataset (MultiChannelEEGDataset): The dataset.
        n_folds (Optional[int]): Max number of folds to generate.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (train_idx, test_idx).
    """
    patients = np.array(dataset.patient_ids)
    unique_patients = np.unique(patients)
    
    splits = []
    for test_patient in unique_patients:
        test_mask = (patients == test_patient)
        train_mask = ~test_mask
        
        test_indices = np.where(test_mask)[0]
        train_indices = np.where(train_mask)[0]
        
        splits.append((train_indices, test_indices))
        
        if n_folds is not None and len(splits) >= n_folds:
            break
            
    return splits
