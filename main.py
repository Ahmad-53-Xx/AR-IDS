import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.train import train_ids_model
from src.attack import generate_adversarial_traffic
from src.defense import adversarial_training

if __name__ == "__main__":
    dataset_path = "data/processed/UNSW_NB15_training-set.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"[-] Error: Could not find {dataset_path}.")
    else:
        # === Phase 3: Baseline Training ===
        print("=== Phase 3: Training Baseline IDS Models ===")
        trained_model, standard_scaler = train_ids_model(dataset_path)
        
        # === Data Prep ===
        df = pd.read_parquet(dataset_path)
        cols_to_drop = ['id', 'proto', 'service', 'state', 'attack_cat']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        X = df.drop(columns=['label']).values
        y = df['label'].values
        
        # Get both train and test splits so we can retrain later
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = standard_scaler.transform(X_train)
        X_test_scaled = standard_scaler.transform(X_test)
        
        # === Phase 4: The Attack ===
        print("\n=== Phase 4: Executing Adversarial Evasion Attack ===")
        # Capture the generated adversarial data this time!
        adv_traffic, _ = generate_adversarial_traffic(trained_model, standard_scaler, X_test_scaled, y_test, epsilon=0.2)
        
        # Isolate labels for the adversarial traffic (they are all malicious, so label = 1)
        malicious_indices = np.where(y_test == 1)[0]
        adv_labels = y_test[malicious_indices]
        
        # === Phase 5: The Defense ===
        # Pass the extracted training data and the new attack data to our hardener
        hardened_model = adversarial_training(trained_model, X_train_scaled, y_train, adv_traffic, adv_labels)
        
        # === Phase 6: Verification ===
        print("\n=== Phase 6: Verifying the Hardened Defender ===")
        print("[*] Re-launching the exact same attack against the hardened model...")
        # We run the attack function again, but test the HARDENED model to see if it survives
        generate_adversarial_traffic(hardened_model, standard_scaler, X_test_scaled, y_test, epsilon=0.2)
