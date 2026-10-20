import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def adversarial_training(model, X_train, y_train, X_adv, y_adv, batch_size=256, epochs=10):
    """
    Retrains the model by injecting adversarial examples into the training set.
    This forces the model to learn the 'shape' of camouflaged traffic.
    """
    print("\n=== Phase 5: Hardening the AI (Adversarial Retraining) ===")
    print("[*] Merging standard traffic with camouflaged attack data...")
    
    # 1. Combine normal training data with the new malicious camouflaged data
    X_combined = np.vstack((X_train, X_adv))
    y_combined = np.concatenate((y_train, y_adv))
    
    # 2. Convert to PyTorch tensors
    dataset = TensorDataset(torch.FloatTensor(X_combined), torch.FloatTensor(y_combined).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Retrain the model to learn the new patterns
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower Learning Rate for fine-tuning
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"    Hardening Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
            
    print("[+] Model hardened successfully.")
    return model
