# ⚛️ QuantumBottleneckAutoencoder (HQAE) & Q-LOCK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35+-green.svg)](https://pennylane.ai/)
[![PhysioNet MIT-BIH](https://img.shields.io/badge/PhysioNet-MIT--BIH%20ECG-brightgreen.svg)](https://physionet.org/content/mitdb/1.0.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Q-LOCK](https://img.shields.io/badge/Paper-Q--LOCK%20Preprint-purple.svg)](PAPER_QLOCK.md)

> **Hybrid Quantum-Classical Autoencoder** featuring 1D-CNN feature extraction, an 8-Qubit Entangled Variational Quantum Circuit (VQC), and **Q-LOCK** in-circuit $SU(2)^{\otimes N}$ unitary encryption for secure, high-compression biometric telemetry.

---

## 📌 Scientific Paper
📄 **Read the complete scientific paper:** [**Q-LOCK: Secure Quantum-Enhanced Medical Telemetry via Hybrid 1D-CNN Variational Autoencoders and $SU(2)^{\otimes N}$ Unitary Scrambling**](PAPER_QLOCK.md)

---

## 🏥 Clinical Validation: Real ECG Data (PhysioNet MIT-BIH)

We benchmarked the **Quantum 1D-CNN Autoencoder** on real patient electrocardiograms from the **MIT-BIH Arrhythmia Database** (Records `100`, `101`, `102`, `103`) using R-peak aligned single-beat cycles ($N_{in} = 256$ samples, $\sim 0.71$s) compressed into **8 Qubits**:

| Metric | Measured Value (Test Set) | Clinical Standard |
| :--- | :---: | :--- |
| **Pearson Correlation ($r$)** | **$0.9850 \pm 0.0178$** | ⭐ **$98.5\%$ morphological fidelity** |
| **Signal-to-Noise Ratio (SNR)** | **$25.17 \pm 3.65\text{ dB}$** | ⭐ **High-resolution clean reconstruction** |
| **PRD (Percentage Distorsion)** | **$6.07 \pm 3.04\%$** | ⭐ **"Good / Acceptable" (Zigel IEEE standard)** |
| **Mean Squared Error (MSE)** | **$2.12 \times 10^{-3}$** | ⭐ **Negligible reconstruction error** |
| **Compression Ratio (CR)** | **$32:1$ ($96.88\%$)** | ⭐ **256 samples $\to$ 8 Qubits** |

### 📊 Validation & Cryptographic Security Figures:
![ECG Validation and Q-LOCK Security Benchmark](grafico_validacao_ecg.png)

---

## 🔐 Q-LOCK: In-Circuit Unitary Cryptography

**Q-LOCK** embeds a 24-dimensional continuous secret key $\vec{K} \in [0, 2\pi]^{24}$ directly within the Hilbert space of the variational circuit:

$$\text{Alice: } |\psi_{\text{cipher}}\rangle = U(\vec{K}) |\psi(x)\rangle$$
$$\text{Bob: } U^\dagger(\vec{K}) |\psi_{\text{cipher}}\rangle = U^\dagger(\vec{K}) U(\vec{K}) |\psi(x)\rangle = \mathbf{I} |\psi(x)\rangle$$

* **Bob (Authorized Recipient):** Recovers the cardiac P-QRS-T complex with $\text{MSE} = 0.0137$.
* **Eve (Eavesdropper / MitM):** Suffers complete phase scrambling ($\text{MSE} = 0.0509$), preventing biometric feature extraction and ensuring patient privacy (HIPAA / GDPR / LGPD compliance).

```
                                      ALICE (Sender / Hospital)
                               ┌────────────────────────────────────────┐
                               │  1. Real ECG Beat (256 pts, R-aligned)  │
                               │                   │                    │
                               │                   ▼                    │
                               │         1D-CNN Encoder Network         │
                               │                   │                    │
                               │                   ▼                    │
                               │       8 Latent Angles θ ∈ [-π, π]      │
                               │                   │                    │
                               │                   ▼                    │
                               │         AngleEmbedding_Y(θ)            │
                               │                   │                    │
                               │                   ▼                    │
                               │       Model Ansatz W (2 Layers)        │
                               │                   │                    │
                               │                   ▼                    │
                               │      Q-LOCK Cipher: U(Secret Key)      │
                               └───────────────────┬────────────────────┘
                                                   │
                                     QUANTUM TRANSMISSION CHANNEL
                                                   │
                               ┌───────────────────▼────────────────────┐
                               │      Q-LOCK Decipher: U†(Secret Key)   │
                               │                   │                    │
                               │                   ▼                    │
                               │         ⟨PauliZ⟩ Measurement x 8       │
                               │                   │                    │
                               │                   ▼                    │
                               │     1D-ConvTranspose Decoder Network   │
                               │                   │                    │
                               │                   ▼                    │
                               │  Reconstructed ECG Beat (256 pts)      │
                               └────────────────────────────────────────┘
                                      BOB (Recipient / Cardiologist)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/MMontanhes/quantum-bottleneck-autoencoder.git
cd quantum-bottleneck-autoencoder
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Real ECG Validation & Q-LOCK (PhysioNet MIT-BIH)
```bash
python validate_real_ecg.py
```
Outputs `grafico_validacao_ecg.png` showing single-beat overlays, multi-patient test generalization, loss convergence, clinical metric tables, and Alice-Bob-Eve cryptographic analysis.

### 4. Run Classical vs. Quantum Benchmark
```bash
python benchmark.py
```
Outputs `grafico_benchmark.png` comparing clean signal convergence and denoising resilience.

---

## 🔬 Mathematical Formulation

1. **State Encoding**:
   $$|\psi_0\rangle = \bigotimes_{j=0}^{7} R_y(\theta_j) |0\rangle, \quad \theta_j = \pi \cdot \tanh(z_j^{\text{classical}})$$

2. **Entangled Variational Ansatz**:
   $$U(\vec{\phi}) = \prod_{l=1}^{L} \left( \prod_{j=0}^{7} \text{CNOT}_{(j, (j+1)\%8)} \cdot \bigotimes_{j=0}^{7} R(\alpha_{l,j}, \beta_{l,j}, \gamma_{l,j}) \right)$$

3. **Q-LOCK Unitary Rotation**:
   $$U_K(\vec{K}) = \prod_{j=0}^{7} \text{CNOT}_{(j, (j+1)\%8)} \cdot \bigotimes_{j=0}^{7} R(k_{j,0}, k_{j,1}, k_{j,2})$$

4. **Measurement**:
   $$z_j^{\text{quantum}} = \langle \psi_0 | U^\dagger(\vec{\phi}) \sigma_z^{(j)} U(\vec{\phi}) | \psi_0 \rangle \in [-1, 1]$$

---

## 📜 License
MIT License. Free for academic, educational, and commercial research.
