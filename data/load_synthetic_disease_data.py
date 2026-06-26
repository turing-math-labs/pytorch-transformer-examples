"""
Data loader for synthetic disease risk prediction dataset.
Loads and prepares the generated synthetic data for training transformer models.
"""

import json
import os
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader


class DiseaseRiskDataset(Dataset):
    """PyTorch Dataset for disease risk prediction tasks."""
    
    def __init__(self, json_file: str, max_input_length: int = 500, max_output_length: int = 200):
        """
        Args:
            json_file: Path to the JSON file with synthetic data
            max_input_length: Maximum length of input text
            max_output_length: Maximum length of output text
        """
        self.data = []
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        for sample in raw_data:
            self.data.append({
                'id': sample['id'],
                'inputs': sample['inputs'],
                'outputs': sample['outputs'],
                'profile': sample['profile']
            })
        
        print(f"Loaded {len(self.data)} samples from {json_file}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get a single sample."""
        sample = self.data[idx]
        return {
            'id': sample['id'],
            'inputs': sample['inputs'][:self.max_input_length],
            'outputs': sample['outputs'][:self.max_output_length],
            'profile': sample['profile']
        }
    
    def get_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple['DiseaseRiskDataset', 'DiseaseRiskDataset', 'DiseaseRiskDataset']:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion for training (default 0.8)
            val_ratio: Proportion for validation (default 0.1)
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        total = len(self.data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size
        
        train_data = self.data[:train_size]
        val_data = self.data[train_size:train_size + val_size]
        test_data = self.data[train_size + val_size:]
        
        train_dataset = DiseaseRiskDataset.__new__(DiseaseRiskDataset)
        train_dataset.data = train_data
        train_dataset.max_input_length = self.max_input_length
        train_dataset.max_output_length = self.max_output_length
        
        val_dataset = DiseaseRiskDataset.__new__(DiseaseRiskDataset)
        val_dataset.data = val_data
        val_dataset.max_input_length = self.max_input_length
        val_dataset.max_output_length = self.max_output_length
        
        test_dataset = DiseaseRiskDataset.__new__(DiseaseRiskDataset)
        test_dataset.data = test_data
        test_dataset.max_input_length = self.max_input_length
        test_dataset.max_output_length = self.max_output_length
        
        print(f"Split dataset - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    json_file: str,
    batch_size: int = 32,
    max_input_length: int = 500,
    max_output_length: int = 200,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        json_file: Path to the JSON file with synthetic data
        batch_size: Batch size for DataLoaders
        max_input_length: Maximum length of input text
        max_output_length: Maximum length of output text
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = DiseaseRiskDataset(json_file, max_input_length, max_output_length)
    train_data, val_data, test_data = dataset.get_split()
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def print_sample_data(json_file: str, num_samples: int = 3) -> None:
    """Print sample data from the dataset for inspection."""
    dataset = DiseaseRiskDataset(json_file)
    
    print(f"\n{'='*80}")
    print(f"Dataset Statistics")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n{'='*80}")
        print(f"Sample {i + 1} (ID: {sample['id']})")
        print(f"{'='*80}")
        
        input_lines = sample['inputs'].split('\n')
        output_lines = sample['outputs'].split('\n')
        
        print(f"\nINPUTS ({len(input_lines)} sentences):")
        print("-" * 80)
        print(sample['inputs'])
        
        print(f"\nOUTPUTS ({len(output_lines)} sentences):")
        print("-" * 80)
        print(sample['outputs'])
        
        print(f"\nPROFILE:")
        print("-" * 80)
        for key, value in sample['profile'].items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    json_file = os.path.join('data', 'synthetic_disease_data.json')
    
    if os.path.exists(json_file):
        print_sample_data(json_file, num_samples=3)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            json_file,
            batch_size=16,
            max_input_length=500,
            max_output_length=200 )
        
        print(f"\n{'='*80}")
        print(f"DataLoader Statistics")
        print(f"{'='*80}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    else:
        print(f"Dataset file '{json_file}' not found.")
        print("Please run 'generate_synthetic_disease_data.py' first to create the dataset.")
