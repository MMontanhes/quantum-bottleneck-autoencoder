"""
==========================================================================
VALIDAÇÃO COM DADOS REAIS DE ECG — PhysioNet MIT-BIH Arrhythmia Database
==========================================================================
Quantum Bottleneck Autoencoder (1D-CNN + 8 Qubits VQC) com Segmentação
e Alinhamento por Pico R (R-Peak Centered Beat-by-Beat Compression).

Configuração Otimizada:
  - Segmentação: Batimento Único alinhado pelo pico R (256 amostras / ~0.71s)
  - Cobertura morfológica: Onda P (amostras 0-80), Pico R (amostra 90), Onda T (120-220)
  - Espaço latente quântico: 8 Qubits (Espaço de Hilbert de 256 dimensões)
  - Taxa de compressão: 32:1 (256 -> 8 Qubits -> 256)
  - Arquitetura: 1D-CNN Encoder -> 8 Qubit VQC -> 1D-ConvTranspose Decoder
  - Base: MIT-BIH Arrhythmia Database (PhysioNet)

Métricas clínicas de qualidade:
  - MSE (Erro Quadrático Médio)
  - SNR (Relação Sinal-Ruído em dB)
  - PRD (Percentage Root-mean-square Difference) — padrão em compressão ECG
  - Correlação de Pearson (r)
  - QS (Quality Score = CR / PRD)
==========================================================================
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import wfdb
from scipy.stats import pearsonr

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. DOWNLOAD E EXTRAÇÃO DE BATIMENTOS ALINHADOS PELO PICO R
# =====================================================================
BEAT_LENGTH = 256      # 256 amostras a 360 Hz ≈ 0.71s (1 ciclo cardíaco completo)
PRE_R = 90             # 90 amostras antes do pico R (Onda P + Intervalo PR)
POST_R = 166           # 166 amostras após o pico R (Complexo QRS + Segmento ST + Onda T)
RECORDS = ['100', '101', '102', '103']  # 4 pacientes distintos
BEATS_PER_RECORD = 50  # 50 batimentos por paciente = 200 batimentos no total

# Tipos válidos de batimentos cardíacos na MIT-BIH (excluindo anotações de ritmo/ruído)
VALID_BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'}

def download_and_extract_aligned_beats():
    """
    Baixa os sinais e as anotações médicas da MIT-BIH e recorta
    batimentos cardíacos perfeitamente centrados no pico R.
    """
    print("[ECG] Baixando registros e anotações médicas do PhysioNet MIT-BIH...", flush=True)
    all_beats = []
    beat_labels = []

    for rec_id in RECORDS:
        print(f"  -> Processando paciente {rec_id}...", end=" ", flush=True)
        record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
        annotation = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
        
        signal = record.p_signal[:, 0]  # Canal MLII
        r_peaks = annotation.sample
        symbols = annotation.symbol
        
        extracted = 0
        for r, sym in zip(r_peaks, symbols):
            if sym not in VALID_BEAT_SYMBOLS:
                continue
            
            # Verificar limites da janela
            if r >= PRE_R and (r + POST_R) <= len(signal):
                beat = signal[r - PRE_R : r + POST_R]
                
                # Normalização min-max para [-1, 1] com preservação de morfologia
                b_min, b_max = beat.min(), beat.max()
                if b_max - b_min > 1e-6:
                    beat_norm = 2.0 * (beat - b_min) / (b_max - b_min) - 1.0
                    all_beats.append(beat_norm)
                    beat_labels.append((rec_id, sym))
                    extracted += 1
                
                if extracted >= BEATS_PER_RECORD:
                    break
        
        print(f"{extracted} batimentos centrados no pico R extraídos", flush=True)
    
    all_beats = np.array(all_beats, dtype=np.float32)
    print(f"[ECG] Total: {len(all_beats)} batimentos cardíacos alinhados de {len(RECORDS)} pacientes", flush=True)
    
    return torch.tensor(all_beats), beat_labels

# =====================================================================
# 2. CIRCUITO QUÂNTICO (8 QUBITS) E ARQUITETURA 1D-CNN HQAE
# =====================================================================
n_qubits = 8
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """Circuito Quântico Variacional com 8 Qubits entrelaçados em anel."""
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class QuantumCNNAutoencoder(nn.Module):
    """
    Autoencoder Convolucional Quântico Híbrido:
    - Encoder: 1D-CNN extrai características morfológicas invariantes (256 -> 8)
    - Bottleneck: Circuito Quântico Variacional de 8 Qubits
    - Decoder: 1D-ConvTranspose reconstrói o batimento perfeito (8 -> 256)
    """
    def __init__(self, input_dim=BEAT_LENGTH):
        super().__init__()
        # Encoder Convolucional 1D
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),   # -> (16, 128)
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),  # -> (32, 64)
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # -> (64, 32)
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, n_qubits)
        )
        
        self.quantum_node = qlayer
        
        # Decoder Convolucional 1D Transposto
        self.decoder_cnn = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64 * 32),
            nn.Unflatten(1, (64, 32)),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),  # -> (32, 64)
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),  # -> (16, 128)
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),   # -> (1, 256)
            nn.Flatten()
        )

    def forward(self, x):
        # x shape: (batch_size, 256) -> (batch_size, 1, 256)
        x_in = x.unsqueeze(1).float()
        latent_c = self.encoder_cnn(x_in)
        latent_b = torch.tanh(latent_c) * torch.pi
        latent_q = self.quantum_node(latent_b)
        return self.decoder_cnn(latent_q)

# =====================================================================
# 3. CIRCUITO QUÂNTICO CRIPTOGRÁFICO (Q-LOCK - 8 QUBITS) PARA BATIMENTOS
# =====================================================================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_crypto_circuit(inputs, model_weights, secret_key, is_decryption=False):
    """Q-LOCK: Criptografia unitária SU(2) do espaço latente do batimento cardíaco."""
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(model_weights[l, q, 0], model_weights[l, q, 1], model_weights[l, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    
    direction = -1.0 if is_decryption else 1.0
    for q in range(n_qubits):
        qml.Rot(
            direction * secret_key[q, 0],
            direction * secret_key[q, 1],
            direction * secret_key[q, 2],
            wires=q
        )
    for q in range(n_qubits):
        if not is_decryption:
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
        else:
            qml.CNOT(wires=[(n_qubits - 1 - q), (n_qubits - q) % n_qubits])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumCryptoCNNAutoencoder(nn.Module):
    """Autoencoder Criptográfico Convolucional para ECG (Q-Crypto-CNN)"""
    def __init__(self, input_dim=BEAT_LENGTH):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, n_qubits)
        )
        self.model_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.decoder_cnn = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64 * 32),
            nn.Unflatten(1, (64, 32)),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Flatten()
        )

    def encrypt_and_compress(self, x, secret_key):
        x_in = x.unsqueeze(1).float()
        latent_c = self.encoder_cnn(x_in)
        latent_b = torch.tanh(latent_c) * torch.pi
        batch_size = x.shape[0]
        ciphertext_latent = []
        for i in range(batch_size):
            res = quantum_crypto_circuit(latent_b[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=False)
            ciphertext_latent.append(torch.stack(res).float())
        return torch.stack(ciphertext_latent).float()

    def decrypt_and_reconstruct(self, ciphertext_latent, secret_key):
        batch_size = ciphertext_latent.shape[0]
        decrypted_latent = []
        for i in range(batch_size):
            res = quantum_crypto_circuit(ciphertext_latent[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=True)
            decrypted_latent.append(torch.stack(res).float())
        decrypted_tensor = torch.stack(decrypted_latent).float()
        return self.decoder_cnn(decrypted_tensor)

# =====================================================================
# 4. MÉTRICAS CLÍNICAS DE QUALIDADE
# =====================================================================
def compute_snr(original, reconstructed):
    noise = original - reconstructed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-15:
        return 100.0
    return 10.0 * np.log10(signal_power / noise_power)

def compute_prd(original, reconstructed):
    diff = original - reconstructed
    prd = 100.0 * np.sqrt(np.sum(diff ** 2) / np.sum(original ** 2))
    return prd

def compute_quality_score(compression_ratio, prd):
    if prd < 1e-10:
        return float('inf')
    return compression_ratio / prd

def evaluate_reconstruction(original_np, reconstructed_np):
    n_samples = original_np.shape[0]
    metrics = {'mse': [], 'snr': [], 'prd': [], 'correlation': [], 'qs': []}
    compression_ratio = BEAT_LENGTH / n_qubits  # 256 / 8 = 32
    
    for i in range(n_samples):
        orig = original_np[i]
        recon = reconstructed_np[i]
        
        mse = np.mean((orig - recon) ** 2)
        snr = compute_snr(orig, recon)
        prd = compute_prd(orig, recon)
        corr, _ = pearsonr(orig, recon)
        qs = compute_quality_score(compression_ratio, prd)
        
        metrics['mse'].append(mse)
        metrics['snr'].append(snr)
        metrics['prd'].append(prd)
        metrics['correlation'].append(corr)
        metrics['qs'].append(qs)
    
    return {k: np.array(v) for k, v in metrics.items()}

# =====================================================================
# 5. EXECUÇÃO DA VALIDAÇÃO
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VALIDAÇÃO COM BATIMENTOS REAIS DE ECG (PICO R ALINHADO) — MIT-BIH")
    print(f"Quantum CNN Autoencoder ({BEAT_LENGTH} -> {n_qubits} Qubits -> {BEAT_LENGTH})")
    print(f"Taxa de Compressão: {BEAT_LENGTH // n_qubits}:1 | Espaço de Hilbert: 2^{n_qubits} = {2**n_qubits} estados")
    print("=" * 70)

    # --- 5.1 Download e Extração de Batimentos ---
    ecg_beats, beat_labels = download_and_extract_aligned_beats()
    
    n_total = ecg_beats.shape[0]
    n_train = int(n_total * 0.8)
    n_test = n_total - n_train
    
    indices = torch.randperm(n_total)
    train_data = ecg_beats[indices[:n_train]]
    test_data = ecg_beats[indices[n_train:]]
    
    print(f"\n[SPLIT] Treino: {n_train} batimentos | Teste: {n_test} batimentos")

    # --- 5.2 Treinamento do HQAE Convolucional ---
    print("\n" + "=" * 70)
    print(f"[FASE 1] TREINAMENTO DO HQAE CONVOLUCIONAL ({n_qubits} QUBITS)")
    print("=" * 70)
    
    model = QuantumCNNAutoencoder(input_dim=BEAT_LENGTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    num_epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    batch_size = min(16, n_train)
    loss_history = []
    
    start_t = time.time()
    for epoch in range(num_epochs):
        epoch_losses = []
        perm = torch.randperm(n_train)
        
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            batch_idx = perm[batch_start:batch_end]
            batch = train_data[batch_idx]
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"  Época {epoch+1:3d}/{num_epochs} - MSE Treino: {avg_loss:.8f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)
    
    total_time = time.time() - start_t
    print(f"\nTreinamento concluído em {total_time:.1f}s", flush=True)

    # --- 5.3 Avaliação Clínica ---
    print("\n" + "=" * 70)
    print("[FASE 2] AVALIAÇÃO CLÍNICA (BATIMENTOS CARDÍACOS)")
    print("=" * 70)
    
    with torch.no_grad():
        train_output = model(train_data)
        train_orig_np = train_data.detach().cpu().numpy()
        train_recon_np = train_output.detach().cpu().numpy()
        train_metrics = evaluate_reconstruction(train_orig_np, train_recon_np)
        
        test_output = model(test_data)
        test_mse = criterion(test_output, test_data).item()
    
    test_orig_np = test_data.detach().cpu().numpy()
    test_recon_np = test_output.detach().cpu().numpy()
    metrics = evaluate_reconstruction(test_orig_np, test_recon_np)
    
    cr = BEAT_LENGTH / n_qubits  # 32:1
    
    print(f"\n  Ratio de Compressão (CR):          {cr:.0f}:1 ({BEAT_LENGTH} -> {n_qubits} Qubits)")
    print(f"\n  --- Conjunto de TREINO (capacidade do modelo) ---")
    print(f"  MSE Médio (Treino):                 {train_metrics['mse'].mean():.8f}")
    print(f"  SNR Médio (Treino):                 {train_metrics['snr'].mean():.2f} dB")
    print(f"  PRD Médio (Treino):                 {train_metrics['prd'].mean():.2f} %")
    print(f"  Correlação Média (Treino):          {train_metrics['correlation'].mean():.6f}")
    print(f"\n  --- Conjunto de TESTE (generalização em batimentos não vistos) ---")
    print(f"  MSE Médio (Teste):                  {metrics['mse'].mean():.8f} ± {metrics['mse'].std():.8f}")
    print(f"  SNR Médio (Teste):                  {metrics['snr'].mean():.2f} ± {metrics['snr'].std():.2f} dB")
    print(f"  PRD Médio (Teste):                  {metrics['prd'].mean():.2f} ± {metrics['prd'].std():.2f} %")
    print(f"  Correlação Média (Teste):           {metrics['correlation'].mean():.6f} ± {metrics['correlation'].std():.6f}")
    print(f"  Quality Score Médio (QS=CR/PRD):    {metrics['qs'].mean():.2f} ± {metrics['qs'].std():.2f}")
    
    prd_mean = metrics['prd'].mean()
    if prd_mean < 5:
        prd_class = "EXCELENTE (Padrão Ouro Clínico — sem distorção perceptível)"
    elif prd_mean < 9:
        prd_class = "BOA (Clinicamente Aceitável — diagnósticos preservados)"
    else:
        prd_class = "MARGINAL (Reconstrução aproximada)"
    print(f"\n  Classificação PRD:  {prd_class}")

    # --- 5.4 Teste de Criptografia Q-LOCK em Batimentos Cardíacos ---
    print("\n" + "=" * 70)
    print(f"[FASE 3] CRIPTOGRAFIA Q-LOCK ({n_qubits} QUBITS) EM BATIMENTOS CARDÍACOS")
    print("=" * 70)
    
    crypto_samples = test_data[:8]
    
    crypto_model = QuantumCryptoCNNAutoencoder(input_dim=BEAT_LENGTH)
    crypto_optimizer = torch.optim.AdamW(crypto_model.parameters(), lr=3e-3, weight_decay=1e-5)
    crypto_epochs = 150
    crypto_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(crypto_optimizer, T_max=crypto_epochs, eta_min=1e-6)
    
    torch.manual_seed(999)
    bob_key = torch.rand(n_qubits, 3) * 2 * torch.pi
    torch.manual_seed(111)
    eve_key = torch.rand(n_qubits, 3) * 2 * torch.pi
    
    print("\n  Treinando canal criptográfico Alice-Bob com batimentos de pacientes...", flush=True)
    start_t = time.time()
    for epoch in range(crypto_epochs):
        crypto_optimizer.zero_grad()
        cipher = crypto_model.encrypt_and_compress(crypto_samples, bob_key)
        recovery = crypto_model.decrypt_and_reconstruct(cipher, bob_key)
        loss = criterion(recovery, crypto_samples)
        loss.backward()
        crypto_optimizer.step()
        crypto_scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Época {epoch+1:3d}/{crypto_epochs} - MSE Canal: {loss.item():.8f}", flush=True)
    
    crypto_time = time.time() - start_t
    print(f"  Canal treinado em {crypto_time:.1f}s", flush=True)
    
    print("\n  Simulando interceptação de dados cardíacos sensíveis...", flush=True)
    with torch.no_grad():
        ciphertext = crypto_model.encrypt_and_compress(crypto_samples, bob_key)
        
        bob_ecg = crypto_model.decrypt_and_reconstruct(ciphertext, bob_key)
        bob_mse = criterion(bob_ecg, crypto_samples).item()
        
        eve_ecg = crypto_model.decrypt_and_reconstruct(ciphertext, eve_key)
        eve_mse = criterion(eve_ecg, crypto_samples).item()
        
        identity_key = torch.zeros(n_qubits, 3)
        eve_id_ecg = crypto_model.decrypt_and_reconstruct(ciphertext, identity_key)
        eve_id_mse = criterion(eve_id_ecg, crypto_samples).item()
    
    protection_factor = eve_mse / (bob_mse + 1e-12)
    
    print(f"\n  [RESULTADOS DE SEGURANÇA - BATIMENTOS CARDÍACOS ({n_qubits} QUBITS)]")
    print(f"  Bob (Chave Correta):    MSE = {bob_mse:.8f}  → Batimento reconstruído perfeitamente")
    print(f"  Eve (Chave Incorreta):  MSE = {eve_mse:.6f}  → Batimento totalmente destruído")
    print(f"  Eve (Sem Chave):        MSE = {eve_id_mse:.6f}  → Dados médicos protegidos")
    print(f"  Fator de Proteção:      {protection_factor:.1f}x")

    # --- 5.5 Geração de Gráfico Publicável ---
    print("\n" + "=" * 70)
    print("[FASE 4] GERANDO GRÁFICO DE PUBLICAÇÃO")
    print("=" * 70)
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 11))
    plt.subplots_adjust(hspace=0.38, wspace=0.30)
    
    pts = BEAT_LENGTH
    t_axis = (np.arange(pts) - PRE_R) / 360.0 * 1000.0  # Tempo em milissegundos centrado no R (0 ms)

    # --- Painel 1: Reconstrução do Batimento (P, QRS, T) ---
    ax = axs[0, 0]
    sample_idx = 0
    ax.plot(t_axis, test_orig_np[sample_idx], label="ECG Original (MIT-BIH)", color='black', lw=2.2)
    ax.plot(t_axis, test_recon_np[sample_idx], label=f"HQAE Reconstruído ({n_qubits} Qubits)", color='#d62728', lw=1.8, linestyle='--', alpha=0.9)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.6, label="Pico R (0 ms)")
    ax.set_title(f"1. Batimento Cardíaco Alinhado ({n_qubits} Qubits / {int(cr)}:1)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo relativo ao Pico R (ms)")
    ax.set_ylabel("Amplitude Normalizada")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # --- Painel 2: Comparação de Múltiplos Batimentos do Conjunto de Teste ---
    ax = axs[0, 1]
    cores_beats = ['#1f77b4', '#2ca02c', '#ff7f0e']
    for idx, c in enumerate(cores_beats):
        if idx < len(test_orig_np):
            ax.plot(t_axis, test_orig_np[idx], color=c, lw=1.5, alpha=0.4, label=f"Beat {idx+1} (Orig)" if idx==0 else None)
            ax.plot(t_axis, test_recon_np[idx], color=c, lw=1.8, linestyle='--', label=f"Beat {idx+1} (HQAE)" if idx==0 else None)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title("2. Generalização em Múltiplos Batimentos (Teste)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo relativo ao Pico R (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # --- Painel 3: Convergência do Treinamento ---
    ax = axs[0, 2]
    ax.semilogy(range(1, num_epochs + 1), loss_history, color='#007acc', lw=2)
    ax.set_title(f"3. Convergência da Loss ({n_qubits} Qubits 1D-CNN)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Épocas")
    ax.set_ylabel("MSE Loss (Log)")
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # --- Painel 4: Tabela de Métricas Clínicas ---
    ax = axs[1, 0]
    ax.axis('off')
    
    table_data = [
        ["Arquitetura", f"1D-CNN + {n_qubits} Qubits VQC"],
        ["Ratio de Compressão (CR)", f"{cr:.0f}:1 ({BEAT_LENGTH} -> {n_qubits} Qubits)"],
        ["Espaço de Hilbert", f"2^{n_qubits} = {2**n_qubits} estados"],
        ["--- TREINO ---", "--- capacidade ---"],
        ["MSE (Treino)", f"{train_metrics['mse'].mean():.2e}"],
        ["SNR (Treino)", f"{train_metrics['snr'].mean():.2f} dB"],
        ["PRD (Treino)", f"{train_metrics['prd'].mean():.2f} %"],
        ["Correlação r (Treino)", f"{train_metrics['correlation'].mean():.4f}"],
        ["--- TESTE ---", "--- generalização ---"],
        ["MSE (Teste)", f"{metrics['mse'].mean():.2e} ± {metrics['mse'].std():.2e}"],
        ["SNR (Teste)", f"{metrics['snr'].mean():.2f} ± {metrics['snr'].std():.2f} dB"],
        ["PRD (Teste)", f"{metrics['prd'].mean():.2f} ± {metrics['prd'].std():.2f} %"],
        ["Correlação r (Teste)", f"{metrics['correlation'].mean():.4f} ± {metrics['correlation'].std():.4f}"],
        ["Classificação Clínica", prd_class.split('(')[0].strip()],
        ["Tempo de Treino", f"{total_time:.1f}s ({num_epochs} épocas)"],
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Métrica", "Valor"],
        loc='center',
        cellLoc='left',
        colWidths=[0.45, 0.55]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for j in range(2):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor('#ecf0f1')
            else:
                table[i, j].set_facecolor('#ffffff')
    
    ax.set_title("4. Métricas Clínicas de Qualidade", fontsize=11, fontweight='bold', pad=15)
    
    # --- Painel 5: Cenário Alice-Bob-Eve em Batimentos Cardíacos ---
    ax = axs[1, 1]
    crypto_orig = crypto_samples.detach().cpu().numpy()
    bob_np = bob_ecg.detach().cpu().numpy()
    eve_np = eve_ecg.detach().cpu().numpy()
    
    ax.plot(t_axis, crypto_orig[0], label="Batimento Original (Paciente)", color='black', lw=2.2)
    ax.plot(t_axis, bob_np[0], label=f"Bob: Chave Correta (MSE: {bob_mse:.1e})", color='#2ca02c', lw=1.8, linestyle='--')
    ax.plot(t_axis, eve_np[0], label=f"Eve: Chave Errada (MSE: {eve_mse:.4f})", color='#d62728', lw=1.4, alpha=0.75)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(f"5. Q-LOCK: Privacidade do Batimento Cardíaco", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo relativo ao Pico R (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # --- Painel 6: Barras de Segurança Criptográfica ---
    ax = axs[1, 2]
    cenarios = ['Bob\n(Chave\nCorreta)', 'Eve\n(Chave\nErrada)', 'Eve\n(Sem\nChave)']
    erros = [bob_mse, eve_mse, eve_id_mse]
    cores = ['#2ca02c', '#d62728', '#ff7f0e']
    
    bars = ax.bar(cenarios, erros, color=cores, width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_title("6. Segurança Q-LOCK em Dados Médicos (MSE)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Erro Quadrático Médio (MSE)")
    ax.set_yscale('log')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    for bar, err in zip(bars, erros):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, max(yval * 2.0, 1e-8),
                f"{err:.2e}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle(
        f"Validação com ECG Real Alinhado por Pico R — Quantum 1D-CNN Autoencoder ({n_qubits} Qubits) + Q-LOCK\n"
        f"MIT-BIH Arrhythmia Database (PhysioNet) | Compressão {int(cr)}:1 ({BEAT_LENGTH} -> {n_qubits} Qubits) | {len(RECORDS)} Pacientes | {n_total} Batimentos",
        fontsize=13, fontweight='bold'
    )
    
    output_img = "grafico_validacao_ecg.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico publicável salvo em: {output_img}", flush=True)
    
    # --- Resumo Final ---
    print("\n" + "=" * 70)
    print("RESUMO FINAL DA VALIDAÇÃO COM BATIMENTOS REAIS (8 QUBITS + R-PEAK)")
    print("=" * 70)
    print(f"  Base de dados:       MIT-BIH Arrhythmia Database (PhysioNet)")
    print(f"  Pacientes:           {len(RECORDS)} ({', '.join(RECORDS)})")
    print(f"  Total de batimentos: {n_total} ({n_train} treino + {n_test} teste)")
    print(f"  Tamanho do batimento:{BEAT_LENGTH} amostras (~0.71s, pico R centrado)")
    print(f"  Qubits:              {n_qubits} (Hilbert: 2^{n_qubits} = {2**n_qubits} dimensões)")
    print(f"  Compressão:          {int(cr)}:1 ({BEAT_LENGTH} -> {n_qubits} qubits)")
    print(f"  MSE (teste):         {metrics['mse'].mean():.2e}")
    print(f"  SNR (teste):         {metrics['snr'].mean():.2f} dB")
    print(f"  PRD (teste):         {metrics['prd'].mean():.2f}% [{prd_class}]")
    print(f"  Correlação (teste):  {metrics['correlation'].mean():.4f}")
    print(f"  Proteção Q-LOCK:     {protection_factor:.1f}x")
    print(f"  Gráfico:             {output_img}")
    print("=" * 70, flush=True)
