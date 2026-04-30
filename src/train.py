import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from src.model import LightweightECGNet
from src.utils import set_seed, ECGDataset

def train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=64, lr=1e-3, device='cpu', save_path='results/best_model.pth'):
    set_seed(42)
    
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LightweightECGNet().to(device)
    
    # Using BCEWithLogitsLoss because model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in tqdm(train_loader, desc="Training"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Validation"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
            
    pd.DataFrame(history).to_csv('results/train_log.csv', index=False)
    print("Training complete. Log saved to results/train_log.csv")
    return model
