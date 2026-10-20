import torch
import numpy as np

def fgsm_attack(data, epsilon, data_grad):
    """
    Applies the Fast Gradient Sign Method (FGSM) perturbation.
    Formula: perturbed_data = data + epsilon * sign(gradient)
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed data by adjusting each feature slightly in the direction 
    # that maximizes the model's error
    perturbed_data = data + epsilon * sign_data_grad
    
    return perturbed_data

def generate_adversarial_traffic(model, scaler, X_test, y_test, epsilon=0.1):
    """
    Takes malicious traffic that the model caught, applies mathematical camouflage (FGSM),
    and tests if the model still catches it.
    """
    print(f"\n[*] Launching Adversarial Attack (FGSM) with Epsilon = {epsilon}...")
    
    # Isolate ONLY the malicious traffic (Label == 1)
    malicious_indices = np.where(y_test == 1)[0]
    X_malicious = X_test[malicious_indices]
    y_malicious = y_test[malicious_indices]
    
    # Convert to PyTorch tensors and require gradients (crucial for the attack)
    data = torch.FloatTensor(X_malicious)
    data.requires_grad = True
    target = torch.FloatTensor(y_malicious).unsqueeze(1)
    
    # 1. Forward pass: See how the model performs BEFORE the attack
    model.eval()
    initial_outputs = model(data)
    initial_preds = (initial_outputs >= 0.5).float()
    
    # Calculate how many attacks were caught originally (should be near 99%)
    original_caught = initial_preds.eq(target).sum().item()
    original_detection_rate = original_caught / len(data)
    print(f"    -> Original Detection Rate on this batch: {original_detection_rate * 100:.2f}%")
    
    # 2. Calculate the Gradients (The Attack Math)
    criterion = torch.nn.BCELoss()
    loss = criterion(initial_outputs, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    
    # 3. Apply the FGSM Camouflage
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    
    # 4. Re-evaluate: Feed the camouflaged traffic back into the model
    with torch.no_grad():
        hacked_outputs = model(perturbed_data)
        hacked_preds = (hacked_outputs >= 0.5).float()
        
    # Calculate how many attacks were caught AFTER the camouflage
    hacked_caught = hacked_preds.eq(target).sum().item()
    hacked_detection_rate = hacked_caught / len(data)
    print(f"    -> Adversarial Detection Rate: {hacked_detection_rate * 100:.2f}%")
    
    drop_in_security = (original_detection_rate - hacked_detection_rate) * 100
    print(f"[!] Attack Success! Model security dropped by {drop_in_security:.2f}%\n")
    
    return perturbed_data.detach().numpy(), hacked_detection_rate
