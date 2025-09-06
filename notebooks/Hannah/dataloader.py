import torch
from torch.utils.data import TensorDataset, DataLoader,  WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split


def create_dataloaders_numeric(x_train, y_train, batch_size=32, val_size = 0.2):

    x_train_sample, x_val, y_train_sample, y_val = train_test_split(x_train, y_train, test_size= val_size, random_state=42, stratify=y_train)

    # Daten in PyTorch-Tensoren konvertieren
    X_train_tensor = torch.tensor(x_train_sample.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_sample.values, dtype=torch.float32)
    X_val_tensor   = torch.tensor(x_val.values, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val.values, dtype=torch.float32)

    
    # Sicherstellen, dass die Labels die richtige Dimension für die BCELoss haben
    if y_train_tensor.ndim == 1:
        y_train_tensor = y_train_tensor.unsqueeze(1)
    if y_val_tensor.ndim == 1:
        y_val_tensor = y_val_tensor.unsqueeze(1)

    # Datasets erstellen
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    


    # WeightedRandomSampler für ausgeglichene Klassen in  Training- Batches
    class_counts = torch.bincount(y_train_tensor.squeeze().long())
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[y_train_tensor.squeeze().long()]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # Dataloader 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader