import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from src.models import TrilayeredIDSNetwork, WideIDSNetwork

def compute_security_metrics(y_true, y_pred_probs):
    """Calculates specific metrics crucial for operational network defense."""
    y_pred_classes = (y_pred_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred_classes)
    auc = roc_auc_score(y_true, y_pred_probs)
    
    # Extract confusion matrix to calculate False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall / Detection Rate
    
    return {"Accuracy": acc, "AUC": auc, "FPR (False Alarms)": fpr, "Recall (Detection)": tpr}

   
    # Assuming 'label' is the target column (0 = normal, 1 = attack)
    X = df.drop(columns=['label', 'src_ip', 'dst_ip'], errors='ignore').values
    y = df['label'].values
    
    # Split and Normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Instantiate Trilayered Baseline Model
    input_dim = X_train.shape[1]
    model = TrilayeredIDSNetwork(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    print(f"[*] Training baseline IDS model on {input_dim} features...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
            
    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        test_x = torch.FloatTensor(X_test)
        test_preds = model(test_x).numpy()
        
    metrics = compute_security_metrics(y_test, test_preds)
    print("\n[+] Baseline Training Phase Metrics Evaluated:")
    for metric_name, value in metrics.items():
        print(f"    -> {metric_name}: {value:.4f}")
        
    return model, scaler

def train_ids_model(dataset_path, epochs=15, batch_size=256):
    print(f"[*] Loading dataset from {dataset_path}...")
    
    # 1. Load Parquet Dataset instead of CSV
    df = pd.read_parquet(dataset_path)
    
    # 2. UNSW-NB15 Specific Cleanup
    # Drop categorical/text columns and the multi-class label ('attack_cat') 
    # so we only predict the binary 'label' (0 = normal, 1 = attack)
    cols_to_drop = ['id', 'proto', 'service', 'state', 'attack_cat']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # Split and Normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Instantiate Baseline Model
    input_dim = X_train.shape[1]
    model = TrilayeredIDSNetwork(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    print(f"[*] Training baseline IDS model on {input_dim} features...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
            
    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        test_x = torch.FloatTensor(X_test)
        test_preds = model(test_x).numpy()
        
    metrics = compute_security_metrics(y_test, test_preds)
    print("\n[+] Baseline Training Phase Metrics Evaluated:")
    for metric_name, value in metrics.items():
        print(f"    -> {metric_name}: {value:.4f}")
        
    return model, scaler
