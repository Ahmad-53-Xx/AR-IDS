# 🛡️ AR-IDS: Adversarial-Resistant Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg?style=for-the-badge&logo=pytorch)
![Cybersecurity](https://img.shields.io/badge/Domain-Cybersecurity-10b981.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success.svg?style=for-the-badge)

> **"Hackers no longer attack the code—they attack the math."**

AR-IDS is an end-to-end cybersecurity research framework that demonstrates the severe vulnerability of standard Artificial Intelligence firewalls to mathematical evasion, and provides a fully functional defense mechanism to patch it. 

This project builds a **Deep Learning Intrusion Detection System**, acts as a Red Team to successfully **blind the AI** using gradient-based attacks, and finally acts as a Blue Team to **harden the network** via adversarial retraining.

---

## 🧠 Core Concept & The Threat Vector

Modern Intrusion Detection Systems (IDS) rely heavily on Deep Neural Networks (DNNs) to filter network traffic. While these models report 99%+ accuracy on standard datasets (like UNSW-NB15), they possess a critical blind spot: **Constrained Adversarial Perturbations**.

By applying the **Fast Gradient Sign Method (FGSM)**, an attacker can calculate the exact decision boundary of the AI and inject micro-adjustments into continuous network features (like inter-arrival times or packet sizes). These changes do not break the malicious payload, but they mathematically camouflage the attack, allowing it to bypass the firewall completely. 

**AR-IDS** proves this vulnerability and implements a robust defense to neutralize it.

---

## 📊 Performance Metrics & Proof of Concept

This project evaluates the IDS across three distinct phases. The metrics below demonstrate the catastrophic failure of standard AI under attack, and the successful recovery of the hardened defender.

| Experimental Phase | Overall Accuracy | False Positives | Detection Rate (Recall) | System Status |
| :--- | :---: | :---: | :---: | :--- |
| **1. Baseline Model** *(Standard Traffic)* | 93.32% | 19.75% | **99.44%** | 🟢 Secure (Illusion) |
| **2. Under FGSM Attack** *(Evasion Traffic)* | N/A | N/A | **55.41%** | 🔴 **Compromised** |
| **3. Hardened AR-IDS** *(Evasion Traffic)* | 94.10% | 18.20% | **97.56%** | 🛡️ **Secured** |

*Under a mathematical evasion attack ($\epsilon=0.2$), the baseline firewall's detection plummeted by 44%. After adversarial retraining, the AR-IDS recovered near-perfect threat interception.*

---

## 🏗️ Architecture Pipeline

1. **Data Engineering:** Extracts 32 continuous, behavior-focused network flow features from raw PCAPs/UNSW-NB15 dataset.
2. **Baseline Formulation:** Trains a Trilayered Deep Neural Network optimized via Binary Cross-Entropy (BCE) loss.
3. **Exploit Engine:** Generates functionality-preserving adversarial network traffic using mathematically constrained FGSM.
4. **Defense Mechanism:** Executes an adversarial retraining loop, feeding the camouflaged traffic back into the model to warp the decision boundary and develop mathematical immunity.

---

## ⚙️ Installation & Usage

### Prerequisites
* Linux/Unix Environment (Kali Linux recommended for PCAP generation)
* Python 3.10+

### Quick Start
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ahmad-53-Xx/AR-IDS.git](https://github.com/Ahmad-53-Xx/AR-IDS.git)
   cd AR-IDS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide the Dataset:**
   * Download the `UNSW_NB15_training-set.parquet` dataset from Kaggle.
   * Place the file into the `data/processed/` directory.

4. **Execute the Pipeline:**
   Run the master orchestrator to train the baseline, launch the attack, and deploy the defense.
   ```bash
   python3 main.py
   ```

---

## 📂 Project Structure

```text
AR-IDS/
├── main.py                # Core orchestrator executing the 3-phase pipeline
├── requirements.txt       # Project dependencies
├── data/                  
│   └── processed/         # Target directory for the UNSW-NB15 .parquet dataset
├── docs/
│   └── AR-IDS_Report.tex  # Formal research paper, mathematical proofs, and TikZ diagrams
└── src/
    ├── features.py        # Scapy-based network parser and feature extraction
    ├── models.py          # PyTorch DNN topology definitions
    ├── train.py           # Baseline training and security metric evaluation
    ├── attack.py          # Fast Gradient Sign Method (FGSM) evasion logic
    └── defense.py         # Adversarial retraining engine
```

---

## 🚀 Future Roadmap
- [ ] Implement Projected Gradient Descent (PGD) as a secondary, iterative attack vector.
- [ ] Develop an Ensemble Defense Architecture (Deep NN + Wide NN + Random Forest) to force consensus voting and suppress the baseline 19% False Positive Rate.
- [ ] Add live PCAP capture streaming directly to the `features.py` parser for real-time defense.

---

**Author:** Ahmad Abu Zayed  
*Independent Cybersecurity & Computer Science Research*
