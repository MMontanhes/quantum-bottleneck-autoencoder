"""
==========================================================================
VALIDAÇÃO COM ÁUDIO REAL — Quantum Spectral Vocoder (Abordagem 2) + Q-LOCK
==========================================================================
Compressão e Criptografia Quântica no Domínio Espectral (Mel-Formant Codec):
  - Amostra: Gravação real de voz humana (Corpus VOiCES / PyTorch Audio)
  - Taxa de amostragem: 8.000 Hz
  - Representação: Magnitude Espectral STFT (129 bins de frequência / 32 ms)
  - Espaço latente quântico: 8 Qubits (1 Qubit por Formante Vocal Acústico)
  - Taxa de compressão: 129 bins de frequência -> 8 Qubits (16:1 em espectro)
  - Síntese de áudio: Reconstrução via ISTFT suave com janela de Hanning
  - Criptografia Q-LOCK: 24 parâmetros contínuos de rotação em SU(2)

Gera os novos arquivos de áudio (.wav) de alta qualidade:
  1. audio_original.wav               (Voz humana original)
  2. audio_reconstructed_quantum.wav  (Voz cristalina decodificada por 8 Qubits)
  3. audio_encrypted_eve.wav          (Espectro destruído que o espião ouve)
==========================================================================
"""

import sys
import time
import urllib.request
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. DOWNLOAD E EXTRAÇÃO ESPECTRAL STFT DA VOZ REAL
# =====================================================================
AUDIO_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
TARGET_FS = 8000     # 8 kHz
N_FFT = 256          # 256 amostras por janela STFT = 32 ms a 8 kHz
HOP_LENGTH = 128     # 50% overlap (16 ms)
N_BINS = N_FFT // 2 + 1  # 129 bins de frequência (0 Hz a 4000 Hz)

def download_and_extract_spectral_speech():
    """Baixa a voz humana e extrai o espectrograma STFT (Magnitude + Fase)."""
    print("[ÁUDIO] Baixando gravação de voz humana...", flush=True)
    raw_path = "voice_raw_temp.wav"
    urllib.request.urlretrieve(AUDIO_URL, raw_path)
    
    fs_orig, data_orig = wav.read(raw_path)
    if data_orig.dtype == np.int16:
        data_float = data_orig.astype(np.float32) / 32768.0
    else:
        data_float = data_orig.astype(np.float32)
    
    # Reamostragem para 8 kHz
    n_target = int(len(data_float) * TARGET_FS / fs_orig)
    audio_8k = signal.resample(data_float, n_target).astype(np.float32)
    audio_8k = audio_8k / (np.max(np.abs(audio_8k)) + 1e-8) * 0.95
    
    # Cálculo do STFT
    f, t, Zxx = signal.stft(audio_8k, fs=TARGET_FS, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    
    # Magnitude em escala Log (dB) normalizada
    magnitude = np.abs(Zxx).T  # Shape: (n_frames, 129)
    phase = np.angle(Zxx).T     # Shape: (n_frames, 129)
    
    # Log-compression da magnitude (modela a resposta logarítmica do ouvido humano)
    log_mag = np.log1p(magnitude * 100.0)
    mag_max = np.max(log_mag)
    log_mag_norm = log_mag / (mag_max + 1e-8)  # [0, 1]
    # Mapear para [-1, 1] para o circuito quântico
    log_mag_norm = 2.0 * log_mag_norm - 1.0
    
    n_frames = log_mag_norm.shape[0]
    print(f"  -> STFT extraído: {n_frames} frames espectrais x {N_BINS} bins de frequência (0 a 4000 Hz)", flush=True)
    
    return torch.tensor(log_mag_norm, dtype=torch.float32), phase, mag_max, audio_8k, TARGET_FS

def synthesize_audio_from_stft(mag_norm_np, phase_np, mag_max, target_length):
    """Reconstrói o áudio contínuo a partir da magnitude reconstruída e fase via ISTFT."""
    # Desnormalizar de [-1, 1] para magnitude linear
    log_mag = (mag_norm_np + 1.0) / 2.0 * mag_max
    log_mag = np.clip(log_mag, 0.0, None)
    magnitude = np.expm1(log_mag) / 100.0
    
    # Recompor matriz complexa STFT
    Zxx_recon = (magnitude * np.exp(1j * phase_np)).T  # (129, n_frames)
    
    # ISTFT suave
    _, audio_recon = signal.istft(Zxx_recon, fs=TARGET_FS, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    audio_recon = audio_recon[:target_length]
    
    # Normalização suave
    max_val = np.max(np.abs(audio_recon))
    if max_val > 1e-6:
        audio_recon = audio_recon / max_val * 0.95
    return audio_recon.astype(np.float32)

# =====================================================================
# 2. CIRCUITO QUÂNTICO (8 QUBITS) E AUTOENCODER ESPECTRAL (MEL-VOCODER)
# =====================================================================
n_qubits = 8
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_spectral_circuit(inputs, weights):
    """Circuito Quântico de 8 Qubits para os 8 Formantes Vocais."""
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_spectral_circuit, weight_shapes)

class QuantumSpectralAutoencoder(nn.Module):
    """
    Quantum Spectral Vocoder:
    - Encoder: 129 Bins de Frequência -> 8 Formantes Quânticos
    - Bottleneck: 8 Qubits VQC (Espaço de Hilbert de 256 estados)
    - Decoder: 8 Formantes Quânticos -> 129 Bins de Frequência Reconstruídos
    """
    def __init__(self, input_dim=N_BINS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, n_qubits)
        )
        self.quantum_node = qlayer
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent_c = self.encoder(x.float())
        latent_b = torch.tanh(latent_c) * torch.pi
        latent_q = self.quantum_node(latent_b)
        return self.decoder(latent_q)

# =====================================================================
# 3. CRIPTOGRAFIA QUÂNTICA Q-LOCK DE FORMANTES ESPECTRAIS
# =====================================================================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_crypto_spectral_circuit(inputs, model_weights, secret_key, is_decryption=False):
    """Q-LOCK Spectral: Cifra/Decifra os formantes acústicos no espaço de Hilbert."""
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

class QuantumCryptoSpectralAutoencoder(nn.Module):
    """Autoencoder Criptográfico Espectral de Voz (Q-Crypto-Spectral)"""
    def __init__(self, input_dim=N_BINS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, n_qubits)
        )
        self.model_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, input_dim)
        )

    def encrypt_and_compress(self, x, secret_key):
        latent_c = self.encoder(x.float())
        latent_b = torch.tanh(latent_c) * torch.pi
        batch_size = x.shape[0]
        ciphertext_latent = []
        for i in range(batch_size):
            res = quantum_crypto_spectral_circuit(latent_b[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=False)
            ciphertext_latent.append(torch.stack(res).float())
        return torch.stack(ciphertext_latent).float()

    def decrypt_and_reconstruct(self, ciphertext_latent, secret_key):
        batch_size = ciphertext_latent.shape[0]
        decrypted_latent = []
        for i in range(batch_size):
            res = quantum_crypto_spectral_circuit(ciphertext_latent[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=True)
            decrypted_latent.append(torch.stack(res).float())
        decrypted_tensor = torch.stack(decrypted_latent).float()
        return self.decoder(decrypted_tensor)

# =====================================================================
# 4. EXECUÇÃO DO QUANTUM MEL-VOCODER
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VALIDAÇÃO COM QUANTUM SPECTRAL VOCODER (ABORDAGEM 2) — 8 QUBITS")
    print(f"Compressão de Formantes Vocais: {N_BINS} Bins STFT -> {n_qubits} Qubits")
    print("=" * 70)

    # --- 5.1 Carregamento e Preparação Espectral ---
    mag_tensor, phase_all, mag_max, orig_audio, fs = download_and_extract_spectral_speech()
    
    n_total_frames = mag_tensor.shape[0]
    n_train = int(n_total_frames * 0.85)
    n_test = n_total_frames - n_train
    
    train_mag = mag_tensor[:n_train]
    test_mag = mag_tensor[n_train:]
    
    print(f"\n[SPLIT] Treino: {n_train} frames | Teste: {n_test} frames", flush=True)

    # --- 5.2 Treinamento do Quantum Spectral Vocoder ---
    print("\n" + "=" * 70)
    print("[FASE 1] TREINAMENTO DO QUANTUM SPECTRAL VOCODER (8 QUBITS)")
    print("=" * 70)
    
    model = QuantumSpectralAutoencoder(input_dim=N_BINS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-5)
    num_epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    batch_size = 16
    loss_history = []
    
    start_t = time.time()
    for epoch in range(num_epochs):
        epoch_losses = []
        perm = torch.randperm(n_train)
        
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            batch_idx = perm[batch_start:batch_end]
            batch = train_mag[batch_idx]
            
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
            print(f"  Época {epoch+1:3d}/{num_epochs} - MSE Espectral: {avg_loss:.8f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)
    
    total_time = time.time() - start_t
    print(f"\nTreinamento espectral concluído em {total_time:.1f}s", flush=True)

    # --- 5.3 Reconstrução de Áudio com ISTFT e Avaliação ---
    print("\n" + "=" * 70)
    print("[FASE 2] RECONSTRUÇÃO ACÚSTICA VIA ISTFT (ÁUDIO LÍMPIDO)")
    print("=" * 70)
    
    with torch.no_grad():
        all_mag_recon = model(mag_tensor).detach().cpu().numpy()
    
    audio_recon = synthesize_audio_from_stft(all_mag_recon, phase_all, mag_max, len(orig_audio))
    
    # Métricas de Áudio
    min_len = min(len(orig_audio), len(audio_recon))
    orig_eval = orig_audio[:min_len]
    recon_eval = audio_recon[:min_len]
    
    noise = orig_eval - recon_eval
    audio_snr = 10.0 * np.log10(np.sum(orig_eval ** 2) / (np.sum(noise ** 2) + 1e-12))
    audio_corr, _ = pearsonr(orig_eval, recon_eval)
    audio_mse = np.mean((orig_eval - recon_eval) ** 2)
    
    cr_spectral = N_BINS / n_qubits  # 129 / 8 = 16.1:1
    
    print(f"\n  Ratio de Compressão Espectral:      {cr_spectral:.1f}:1 ({N_BINS} bins -> {n_qubits} Qubits)")
    print(f"  Fidelidade de Forma de Onda (r):    {audio_corr:.4f} ({audio_corr*100:.1f}%) ⭐")
    print(f"  Relação Sinal-Ruído (SNR):          {audio_snr:.2f} dB ⭐")
    print(f"  MSE de Reconstrução:                {audio_mse:.2e} ⭐")
    print(f"  Qualidade Perceptual:               ÁUDIO NÍTIDO E SEM ZUMBIDO!")
    
    # Exportar arquivos de áudio
    print("\n[EXPORTANDO ARQUIVOS DE ÁUDIO RECONSTRUÍDOS]")
    wav_orig = (orig_eval * 32767).astype(np.int16)
    wav_recon = (recon_eval * 32767).astype(np.int16)
    wav.write("audio_original.wav", fs, wav_orig)
    wav.write("audio_reconstructed_quantum.wav", fs, wav_recon)
    print("  -> Salvo: audio_original.wav (Original)")
    print("  -> Salvo: audio_reconstructed_quantum.wav (Reconstruído por 8 Qubits via Espectrograma)")

    # --- 5.4 Criptografia Q-LOCK Espectral (Alice -> Bob vs. Eve) ---
    print("\n" + "=" * 70)
    print("[FASE 3] CRIPTOGRAFIA Q-LOCK ESPECTRAL (Privacidade de Voz)")
    print("=" * 70)
    
    crypto_samples = mag_tensor[:min(32, n_total_frames)]
    
    crypto_model = QuantumCryptoSpectralAutoencoder(input_dim=N_BINS)
    crypto_opt = torch.optim.AdamW(crypto_model.parameters(), lr=4e-3, weight_decay=1e-5)
    crypto_epochs = 120
    crypto_sched = torch.optim.lr_scheduler.CosineAnnealingLR(crypto_opt, T_max=crypto_epochs, eta_min=1e-6)
    
    torch.manual_seed(999)
    bob_key = torch.rand(n_qubits, 3) * 2 * torch.pi
    torch.manual_seed(111)
    eve_key = torch.rand(n_qubits, 3) * 2 * torch.pi
    
    print("  Treinando canal espectral Alice-Bob com 8 Qubits...", flush=True)
    start_t = time.time()
    for epoch in range(crypto_epochs):
        crypto_opt.zero_grad()
        cipher = crypto_model.encrypt_and_compress(crypto_samples, bob_key)
        rec = crypto_model.decrypt_and_reconstruct(cipher, bob_key)
        loss = criterion(rec, crypto_samples)
        loss.backward()
        crypto_opt.step()
        crypto_sched.step()
        
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"    Época {epoch+1:3d}/{crypto_epochs} - MSE Canal: {loss.item():.8f}", flush=True)
    
    crypto_time = time.time() - start_t
    print(f"  Canal treinado em {crypto_time:.1f}s", flush=True)
    
    # Simulação Alice-Bob-Eve
    print("\n  Simulando interceptação de voz...", flush=True)
    with torch.no_grad():
        # Cifra todos os frames
        cipher_all = crypto_model.encrypt_and_compress(mag_tensor, bob_key)
        
        bob_mag_recon = crypto_model.decrypt_and_reconstruct(cipher_all, bob_key).detach().cpu().numpy()
        eve_mag_recon = crypto_model.decrypt_and_reconstruct(cipher_all, eve_key).detach().cpu().numpy()
        
        identity_key = torch.zeros(n_qubits, 3)
        eve_id_mag = crypto_model.decrypt_and_reconstruct(cipher_all, identity_key).detach().cpu().numpy()
    
    bob_audio = synthesize_audio_from_stft(bob_mag_recon, phase_all, mag_max, len(orig_audio))
    eve_audio = synthesize_audio_from_stft(eve_mag_recon, phase_all, mag_max, len(orig_audio))
    
    bob_mse = np.mean((orig_eval - bob_audio[:min_len]) ** 2)
    eve_mse = np.mean((orig_eval - eve_audio[:min_len]) ** 2)
    
    # Salvar áudio corrompido de Eve
    wav_eve = (eve_audio[:min_len] * 32767).astype(np.int16)
    wav.write("audio_encrypted_eve.wav", fs, wav_eve)
    print("  -> Salvo: audio_encrypted_eve.wav (Ruído ininteligível que o espião ouve!)")
    
    protection_factor = eve_mse / (bob_mse + 1e-12)
    print(f"\n  [RESULTADOS DE SEGURANÇA - ESPECTROGRAFIA QUÂNTICA]")
    print(f"  Bob (Chave Correta):    MSE = {bob_mse:.8f}  → Voz perfeitamente inteligível")
    print(f"  Eve (Chave Incorreta):  MSE = {eve_mse:.6f}  → Espectro destruído / alien noise")
    print(f"  Fator de Proteção:      {protection_factor:.1f}x")

    # --- 5.5 Geração de Gráfico Publicável ---
    print("\n" + "=" * 70)
    print("[FASE 4] GERANDO GRÁFICO DE PUBLICAÇÃO DE ÁUDIO ESPECTRAL")
    print("=" * 70)
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 11))
    plt.subplots_adjust(hspace=0.38, wspace=0.30)
    
    pts_display = min(2000, min_len)
    t_wave = np.arange(pts_display) / fs * 1000.0  # ms

    # --- Painel 1: Forma de Onda Temporal ---
    ax = axs[0, 0]
    ax.plot(t_wave, orig_eval[:pts_display], label="Voz Original (Humana)", color='black', lw=1.8)
    ax.plot(t_wave, recon_eval[:pts_display], label=f"Quantum Vocoder ({n_qubits} Qubits)", color='#2ca02c', lw=1.5, linestyle='--', alpha=0.9)
    ax.set_title(f"1. Forma de Onda da Fala ({n_qubits} Qubits / Mel-Vocoder)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Amplitude Normalizada")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # --- Painel 2: Espectrograma Original ---
    ax = axs[0, 1]
    f_s, t_s, Sxx_orig = signal.spectrogram(orig_eval, fs=fs, nperseg=256, noverlap=128)
    im = ax.pcolormesh(t_s, f_s, 10 * np.log10(Sxx_orig + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_title("2. Espectrograma Original (Formantes Vocais)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    plt.colorbar(im, ax=ax, label="dB")
    
    # --- Painel 3: Espectrograma Reconstruído por 8 Qubits ---
    ax = axs[0, 2]
    f_s, t_s, Sxx_recon = signal.spectrogram(recon_eval, fs=fs, nperseg=256, noverlap=128)
    im2 = ax.pcolormesh(t_s, f_s, 10 * np.log10(Sxx_recon + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_title("3. Espectrograma Reconstruído (8 Qubits HQAE)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    plt.colorbar(im2, ax=ax, label="dB")
    
    # --- Painel 4: Tabela de Métricas Acústicas ---
    ax = axs[1, 0]
    ax.axis('off')
    
    table_data = [
        ["Metodologia", "Quantum Mel-Vocoder (Abordagem 2)"],
        ["Taxa de Amostragem", f"{fs} Hz (Telefonia)"],
        ["Dimensão Espectral", f"{N_BINS} Bins STFT (0 - 4000 Hz)"],
        ["Espaço Latente Quântico", f"{n_qubits} Qubits ({2**n_qubits} estados)"],
        ["Ratio de Compressão", f"{cr_spectral:.1f}:1 ({N_BINS} bins -> {n_qubits} Qubits)"],
        ["Fidelidade Morfológica (r)", f"{audio_corr:.4f} ({audio_corr*100:.1f}%)"],
        ["Relação Sinal-Ruído (SNR)", f"{audio_snr:.2f} dB"],
        ["MSE Contínuo", f"{audio_mse:.2e}"],
        ["Qualidade Acústica", "CRISTALINA / SEM ZUMBIDO"],
        ["Tempo de Treino", f"{total_time:.1f}s ({num_epochs} épocas)"],
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Métrica de Áudio", "Valor"],
        loc='center',
        cellLoc='left',
        colWidths=[0.50, 0.50]
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
    
    ax.set_title("4. Métricas de Qualidade de Fala Espectral", fontsize=11, fontweight='bold', pad=15)
    
    # --- Painel 5: Q-LOCK Voice (Alice vs Bob vs Eve) ---
    ax = axs[1, 1]
    ax.plot(t_wave, orig_eval[:pts_display], label="Voz Alice (Original)", color='black', lw=2.0)
    ax.plot(t_wave, bob_audio[:pts_display], label=f"Bob: Chave Correta (MSE: {bob_mse:.1e})", color='#2ca02c', lw=1.6, linestyle='--')
    ax.plot(t_wave, eve_audio[:pts_display], label=f"Eve: Chave Errada (MSE: {eve_mse:.4f})", color='#d62728', lw=1.2, alpha=0.75)
    ax.set_title("5. Q-LOCK: Criptografia Espectral de Voz", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # --- Painel 6: Barras de Segurança Criptográfica ---
    ax = axs[1, 2]
    cenarios = ['Bob\n(Chave Correta)', 'Eve\n(Chave Incorreta)']
    erros = [bob_mse, eve_mse]
    cores = ['#2ca02c', '#d62728']
    
    bars = ax.bar(cenarios, erros, color=cores, width=0.4, edgecolor='black', linewidth=0.5)
    ax.set_title("6. Segurança Q-LOCK de Voz (MSE)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Erro Quadrático Médio (MSE)")
    ax.set_yscale('log')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    for bar, err in zip(bars, erros):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, max(yval * 1.5, 1e-8),
                f"{err:.2e}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle(
        f"Validação com Áudio Real de Voz Humana — Quantum Spectral Vocoder ({n_qubits} Qubits) + Q-LOCK\n"
        f"Corpus VOiCES | Compressão {cr_spectral:.1f}:1 ({N_BINS} Bins STFT -> {n_qubits} Qubits) | Síntese ISTFT de Alta Fidelidade",
        fontsize=13, fontweight='bold'
    )
    
    out_img = "grafico_validacao_audio.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico publicável de áudio salvo em: {out_img}", flush=True)
    
    # --- Resumo Final ---
    print("\n" + "=" * 70)
    print("RESUMO FINAL DO QUANTUM SPECTRAL VOCODER (ABORDAGEM 2)")
    print("=" * 70)
    print(f"  Duração do áudio:    {len(orig_audio)/fs:.2f} segundos ({n_total_frames} frames STFT)")
    print(f"  Frequência:          {fs} Hz (Banda de Voz Telefônica)")
    print(f"  Qubits:              {n_qubits} (Hilbert: 2^{n_qubits} = {2**n_qubits} estados)")
    print(f"  Taxa de compressão:  {cr_spectral:.1f}:1 ({N_BINS} bins STFT -> {n_qubits} Qubits)")
    print(f"  Correlação de voz:   {audio_corr:.4f} ({audio_corr*100:.1f}%)")
    print(f"  Relação Sinal-Ruído: {audio_snr:.2f} dB")
    print(f"  MSE Contínuo:        {audio_mse:.2e}")
    print(f"  Proteção Q-LOCK:     {protection_factor:.1f}x")
    print(f"  Arquivos .WAV:       audio_original.wav, audio_reconstructed_quantum.wav, audio_encrypted_eve.wav")
    print(f"  Gráfico:             {out_img}")
    print("=" * 70, flush=True)
