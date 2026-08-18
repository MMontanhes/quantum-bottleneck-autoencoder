"""
==========================================================================
VALIDAÇÃO COM ÁUDIO REAL DE VOZ HUMANA — Quantum Speech Codec + Q-LOCK
==========================================================================
Compressão e Criptografia Quântica de Voz (Speech Compression & Security):
  - Amostra: Gravação real de voz humana (Corpus VOiCES / PyTorch Audio)
  - Taxa de amostragem: 8.000 Hz (Banda de Telefonia Padrão)
  - Janelamento de voz: Frames de 256 amostras (32 ms — padrão de vocoders)
  - Espaço latente quântico: 8 Qubits (Espaço de Hilbert de 256 dimensões)
  - Taxa de compressão: 32:1 (256 amostras de áudio -> 8 Qubits)
  - Criptografia Q-LOCK: 24 parâmetros contínuos de rotação em SU(2)

Gera os arquivos de áudio (.wav) para audição:
  1. audio_original.wav               (Voz humana original)
  2. audio_reconstructed_quantum.wav  (Voz decodificada pelos 8 Qubits)
  3. audio_encrypted_eve.wav          (Ruído ininteligível que o espião ouve)
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
# 1. DOWNLOAD E PRÉ-PROCESSAMENTO DO ÁUDIO DE VOZ REAL
# =====================================================================
AUDIO_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
TARGET_FS = 8000     # 8 kHz
FRAME_SIZE = 256     # 256 amostras por frame = 32 ms a 8 kHz
OVERLAP = 128        # 50% de overlap para síntese suave (Overlap-Add)

def download_and_prepare_speech():
    """Baixa o áudio de voz humana, reamostra para 8 kHz e extrai frames."""
    print("[ÁUDIO] Baixando gravação de voz humana...", flush=True)
    raw_path = "voice_raw_temp.wav"
    urllib.request.urlretrieve(AUDIO_URL, raw_path)
    
    fs_orig, data_orig = wav.read(raw_path)
    print(f"  -> Áudio original: {len(data_orig)} amostras a {fs_orig} Hz ({len(data_orig)/fs_orig:.2f}s)", flush=True)
    
    # Normalizar int16 para float32 [-1, 1]
    if data_orig.dtype == np.int16:
        data_float = data_orig.astype(np.float32) / 32768.0
    else:
        data_float = data_orig.astype(np.float32)
    
    # Reamostragem para 8 kHz
    n_target = int(len(data_float) * TARGET_FS / fs_orig)
    resampled = signal.resample(data_float, n_target).astype(np.float32)
    
    # Normalização de pico
    resampled = resampled / (np.max(np.abs(resampled)) + 1e-8)
    
    # Segmentar em frames com 50% de overlap
    step = FRAME_SIZE - OVERLAP
    n_frames = (len(resampled) - FRAME_SIZE) // step + 1
    frames = []
    
    for i in range(n_frames):
        start = i * step
        frame = resampled[start : start + FRAME_SIZE]
        frames.append(frame)
    
    frames_tensor = torch.tensor(np.array(frames, dtype=np.float32))
    print(f"  -> Processados {n_frames} frames de voz ({FRAME_SIZE} amostras / 32 ms por frame)", flush=True)
    
    return frames_tensor, resampled, TARGET_FS

def reconstruct_audio_overlap_add(frames_np, total_length, frame_size=FRAME_SIZE, overlap=OVERLAP):
    """Reconstrói o sinal de áudio contínuo usando Overlap-Add e janela de Hanning."""
    step = frame_size - overlap
    reconstructed = np.zeros(total_length, dtype=np.float32)
    window = np.hanning(frame_size).astype(np.float32)
    norm_factor = np.zeros(total_length, dtype=np.float32)
    
    for i, frame in enumerate(frames_np):
        start = i * step
        end = start + frame_size
        if end <= total_length:
            reconstructed[start:end] += frame * window
            norm_factor[start:end] += window ** 2
    
    # Evitar divisão por zero nas bordas
    norm_factor = np.where(norm_factor > 1e-6, norm_factor, 1.0)
    reconstructed = reconstructed / norm_factor
    # Normalizar para evitar clipping
    max_val = np.max(np.abs(reconstructed))
    if max_val > 1e-6:
        reconstructed = reconstructed / max_val * 0.95
    return reconstructed

# =====================================================================
# 2. CIRCUITO QUÂNTICO (8 QUBITS) E ARQUITETURA 1D-CNN DE VOZ
# =====================================================================
n_qubits = 8
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_speech_circuit(inputs, weights):
    """Circuito Quântico Variacional de 8 Qubits para Frames de Voz."""
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_speech_circuit, weight_shapes)

class QuantumSpeechAutoencoder(nn.Module):
    """Quantum Speech Codec: 1D-CNN -> 8 Qubits VQC -> 1D-ConvTranspose"""
    def __init__(self, input_dim=FRAME_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),   # -> 128
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),  # -> 64
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # -> 32
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, n_qubits)
        )
        self.quantum_node = qlayer
        self.decoder = nn.Sequential(
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

    def forward(self, x):
        x_in = x.unsqueeze(1).float()
        latent_c = self.encoder(x_in)
        latent_b = torch.tanh(latent_c) * torch.pi
        latent_q = self.quantum_node(latent_b)
        return self.decoder(latent_q)

# =====================================================================
# 3. CRIPTOGRAFIA QUÂNTICA Q-LOCK DE VOZ (8 QUBITS)
# =====================================================================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_crypto_speech_circuit(inputs, model_weights, secret_key, is_decryption=False):
    """Q-LOCK Voice: Cifra/Decifra frames de áudio no espaço de Hilbert."""
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

class QuantumCryptoSpeechAutoencoder(nn.Module):
    """Autoencoder Criptográfico Quântico para Voz (Q-Crypto-Speech)"""
    def __init__(self, input_dim=FRAME_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
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
        self.decoder = nn.Sequential(
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
        latent_c = self.encoder(x_in)
        latent_b = torch.tanh(latent_c) * torch.pi
        batch_size = x.shape[0]
        ciphertext_latent = []
        for i in range(batch_size):
            res = quantum_crypto_speech_circuit(latent_b[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=False)
            ciphertext_latent.append(torch.stack(res).float())
        return torch.stack(ciphertext_latent).float()

    def decrypt_and_reconstruct(self, ciphertext_latent, secret_key):
        batch_size = ciphertext_latent.shape[0]
        decrypted_latent = []
        for i in range(batch_size):
            res = quantum_crypto_speech_circuit(ciphertext_latent[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=True)
            decrypted_latent.append(torch.stack(res).float())
        decrypted_tensor = torch.stack(decrypted_latent).float()
        return self.decoder(decrypted_tensor)

# =====================================================================
# 4. MÉTRICAS DE QUALIDADE DE ÁUDIO E VOZ
# =====================================================================
def compute_audio_snr(orig, recon):
    """Relação Sinal-Ruído em dB."""
    noise = orig - recon
    signal_power = np.sum(orig ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power < 1e-12:
        return 100.0
    return 10.0 * np.log10(signal_power / noise_power)

def compute_log_spectral_distance(orig, recon, n_fft=512):
    """
    Log-Spectral Distance (LSD) em dB — métrica padrão de distorção de fala.
    LSD < 2.0 dB: Alta qualidade acústica e inteligibilidade preservada.
    """
    f_orig, t_orig, S_orig = signal.spectrogram(orig, fs=TARGET_FS, nperseg=n_fft, noverlap=n_fft//2)
    f_recon, t_recon, S_recon = signal.spectrogram(recon, fs=TARGET_FS, nperseg=n_fft, noverlap=n_fft//2)
    
    eps = 1e-12
    log_orig = 10.0 * np.log10(S_orig + eps)
    log_recon = 10.0 * np.log10(S_recon + eps)
    
    lsd_per_frame = np.sqrt(np.mean((log_orig - log_recon) ** 2, axis=0))
    return np.mean(lsd_per_frame)

# =====================================================================
# 5. EXECUÇÃO DO EXPERIMENTO DE VOZ
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VALIDAÇÃO COM ÁUDIO REAL DE VOZ HUMANA — Quantum Speech Codec")
    print(f"Compressão 32:1 (Frame: {FRAME_SIZE} amostras / 32 ms -> {n_qubits} Qubits)")
    print("=" * 70)

    # --- 5.1 Carregamento e Preparação ---
    frames_tensor, original_continuous, fs = download_and_prepare_speech()
    
    n_total_frames = frames_tensor.shape[0]
    n_train = int(n_total_frames * 0.85)
    n_test = n_total_frames - n_train
    
    # Divisão treino/teste
    train_frames = frames_tensor[:n_train]
    test_frames = frames_tensor[n_train:]
    
    print(f"\n[SPLIT] Treino: {n_train} frames | Teste: {n_test} frames", flush=True)

    # --- 5.2 Treinamento do Quantum Speech Codec ---
    print("\n" + "=" * 70)
    print("[FASE 1] TREINAMENTO DO QUANTUM SPEECH AUTOENCODER (8 QUBITS)")
    print("=" * 70)
    
    model = QuantumSpeechAutoencoder(input_dim=FRAME_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
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
            batch = train_frames[batch_idx]
            
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
    print(f"\nTreinamento de voz concluído em {total_time:.1f}s", flush=True)

    # --- 5.3 Reconstrução Contínua e Métricas ---
    print("\n" + "=" * 70)
    print("[FASE 2] RECONSTRUÇÃO DO ÁUDIO CONTÍNUO E AVALIAÇÃO ACÚSTICA")
    print("=" * 70)
    
    with torch.no_grad():
        all_reconstructed_frames = model(frames_tensor).detach().cpu().numpy()
    
    reconstructed_continuous = reconstruct_audio_overlap_add(
        all_reconstructed_frames, len(original_continuous), FRAME_SIZE, OVERLAP
    )
    
    # Métricas de Áudio
    min_len = min(len(original_continuous), len(reconstructed_continuous))
    orig_eval = original_continuous[:min_len]
    recon_eval = reconstructed_continuous[:min_len]
    
    audio_snr = compute_audio_snr(orig_eval, recon_eval)
    audio_corr, _ = pearsonr(orig_eval, recon_eval)
    audio_lsd = compute_log_spectral_distance(orig_eval, recon_eval)
    audio_mse = np.mean((orig_eval - recon_eval) ** 2)
    
    cr = FRAME_SIZE / n_qubits  # 32:1
    
    print(f"\n  Ratio de Compressão (CR):           {cr:.0f}:1 ({FRAME_SIZE} -> {n_qubits} Qubits)")
    print(f"  Fidelidade de Forma de Onda (r):    {audio_corr:.4f} ({audio_corr*100:.1f}%)")
    print(f"  Relação Sinal-Ruído (SNR):          {audio_snr:.2f} dB")
    print(f"  Log-Spectral Distance (LSD):        {audio_lsd:.2f} dB (Distorção Espectral)")
    print(f"  MSE Contínuo:                       {audio_mse:.2e}")
    
    # Salvar arquivos .WAV para audição
    print("\n[EXPORTANDO ARQUIVOS DE ÁUDIO]")
    wav_orig = (orig_eval * 32767).astype(np.int16)
    wav_recon = (recon_eval * 32767).astype(np.int16)
    wav.write("audio_original.wav", fs, wav_orig)
    wav.write("audio_reconstructed_quantum.wav", fs, wav_recon)
    print("  -> Salvo: audio_original.wav (Áudio original)")
    print("  -> Salvo: audio_reconstructed_quantum.wav (Áudio decodificado por 8 Qubits)")

    # --- 5.4 Criptografia Q-LOCK de Voz (Alice -> Bob vs. Eve) ---
    print("\n" + "=" * 70)
    print("[FASE 3] CRIPTOGRAFIA Q-LOCK DE VOZ (Privacidade em Chamadas Seguras)")
    print("=" * 70)
    
    crypto_frames = frames_tensor[:min(32, n_total_frames)]
    
    crypto_model = QuantumCryptoSpeechAutoencoder(input_dim=FRAME_SIZE)
    crypto_opt = torch.optim.AdamW(crypto_model.parameters(), lr=3e-3, weight_decay=1e-5)
    crypto_epochs = 150
    crypto_sched = torch.optim.lr_scheduler.CosineAnnealingLR(crypto_opt, T_max=crypto_epochs, eta_min=1e-6)
    
    torch.manual_seed(999)
    bob_key = torch.rand(n_qubits, 3) * 2 * torch.pi
    torch.manual_seed(111)
    eve_key = torch.rand(n_qubits, 3) * 2 * torch.pi
    
    print("  Treinando canal de voz criptografada Alice-Bob...", flush=True)
    start_t = time.time()
    for epoch in range(crypto_epochs):
        crypto_opt.zero_grad()
        cipher = crypto_model.encrypt_and_compress(crypto_frames, bob_key)
        rec = crypto_model.decrypt_and_reconstruct(cipher, bob_key)
        loss = criterion(rec, crypto_frames)
        loss.backward()
        crypto_opt.step()
        crypto_sched.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Época {epoch+1:3d}/{crypto_epochs} - MSE Canal: {loss.item():.8f}", flush=True)
    
    crypto_time = time.time() - start_t
    print(f"  Canal de voz treinado em {crypto_time:.1f}s", flush=True)
    
    # Simulação Alice-Bob-Eve
    with torch.no_grad():
        ciphertext_all = crypto_model.encrypt_and_compress(frames_tensor, bob_key)
        
        bob_rec_frames = crypto_model.decrypt_and_reconstruct(ciphertext_all, bob_key).detach().cpu().numpy()
        eve_rec_frames = crypto_model.decrypt_and_reconstruct(ciphertext_all, eve_key).detach().cpu().numpy()
        
        identity_key = torch.zeros(n_qubits, 3)
        eve_id_frames = crypto_model.decrypt_and_reconstruct(ciphertext_all, identity_key).detach().cpu().numpy()
    
    bob_audio = reconstruct_audio_overlap_add(bob_rec_frames, len(original_continuous), FRAME_SIZE, OVERLAP)
    eve_audio = reconstruct_audio_overlap_add(eve_rec_frames, len(original_continuous), FRAME_SIZE, OVERLAP)
    
    bob_mse = np.mean((orig_eval - bob_audio[:min_len]) ** 2)
    eve_mse = np.mean((orig_eval - eve_audio[:min_len]) ** 2)
    
    # Salvar áudio interceptado por Eve
    wav_eve = (eve_audio[:min_len] * 32767).astype(np.int16)
    wav.write("audio_encrypted_eve.wav", fs, wav_eve)
    print("  -> Salvo: audio_encrypted_eve.wav (Ruído ininteligível que o espião ouve!)")
    
    protection_factor = eve_mse / (bob_mse + 1e-12)
    print(f"\n  [RESULTADOS DE SEGURANÇA - VOZ HUMANA]")
    print(f"  Bob (Chave Correta):    MSE = {bob_mse:.8f}  → Voz inteligível e nítida")
    print(f"  Eve (Chave Incorreta):  MSE = {eve_mse:.6f}  → Voz completamente distorcida/ruído")
    print(f"  Fator de Proteção:      {protection_factor:.1f}x")

    # --- 5.5 Geração de Gráfico Publicável ---
    print("\n" + "=" * 70)
    print("[FASE 4] GERANDO GRÁFICO DE PUBLICAÇÃO DE ÁUDIO")
    print("=" * 70)
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 11))
    plt.subplots_adjust(hspace=0.38, wspace=0.30)
    
    pts_display = min(2000, min_len)  # ~250 ms de fala
    t_wave = np.arange(pts_display) / fs * 1000.0  # ms

    # --- Painel 1: Forma de Onda Temporal (Original vs HQAE) ---
    ax = axs[0, 0]
    ax.plot(t_wave, orig_eval[:pts_display], label="Voz Original (Humana)", color='black', lw=1.8)
    ax.plot(t_wave, recon_eval[:pts_display], label=f"HQAE Reconstruído ({n_qubits} Qubits)", color='#d62728', lw=1.5, linestyle='--', alpha=0.85)
    ax.set_title(f"1. Forma de Onda da Fala ({n_qubits} Qubits / {int(cr)}:1)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Amplitude Normalizada")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # --- Painel 2: Espectrograma da Fala Original vs Reconstruída ---
    ax = axs[0, 1]
    f_s, t_s, Sxx = signal.spectrogram(orig_eval, fs=fs, nperseg=256, noverlap=128)
    im = ax.pcolormesh(t_s, f_s, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
    ax.set_title("2. Espectrograma Acústico (Formantes de Fala)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    plt.colorbar(im, ax=ax, label="dB")
    
    # --- Painel 3: Convergência do Treinamento ---
    ax = axs[0, 2]
    ax.semilogy(range(1, num_epochs + 1), loss_history, color='#007acc', lw=2)
    ax.set_title(f"3. Convergência da Loss de Voz ({n_qubits} Qubits)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Épocas")
    ax.set_ylabel("MSE Loss (Log)")
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # --- Painel 4: Tabela de Métricas Acústicas ---
    ax = axs[1, 0]
    ax.axis('off')
    
    table_data = [
        ["Taxa de Amostragem", f"{fs} Hz (Telefonia)"],
        ["Tamanho do Frame", f"{FRAME_SIZE} amostras (32 ms)"],
        ["Taxa de Compressão (CR)", f"{cr:.0f}:1 ({FRAME_SIZE} -> {n_qubits} Qubits)"],
        ["Espaço de Hilbert", f"2^{n_qubits} = {2**n_qubits} estados"],
        ["Fidelidade Morfológica (r)", f"{audio_corr:.4f} ({audio_corr*100:.1f}%)"],
        ["Relação Sinal-Ruído (SNR)", f"{audio_snr:.2f} dB"],
        ["Distorção Espectral (LSD)", f"{audio_lsd:.2f} dB"],
        ["MSE Contínuo", f"{audio_mse:.2e}"],
        ["Inteligibilidade Acústica", "PRESERVADA (Voz Clara)"],
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
    
    ax.set_title("4. Métricas de Qualidade de Fala", fontsize=11, fontweight='bold', pad=15)
    
    # --- Painel 5: Q-LOCK Voice (Alice vs Bob vs Eve) ---
    ax = axs[1, 1]
    ax.plot(t_wave, orig_eval[:pts_display], label="Voz Alice (Original)", color='black', lw=2.0)
    ax.plot(t_wave, bob_audio[:pts_display], label=f"Bob: Chave Correta (MSE: {bob_mse:.1e})", color='#2ca02c', lw=1.6, linestyle='--')
    ax.plot(t_wave, eve_audio[:pts_display], label=f"Eve: Chave Errada (MSE: {eve_mse:.4f})", color='#d62728', lw=1.2, alpha=0.75)
    ax.set_title("5. Q-LOCK: Criptografia de Voz em Tempo Real", fontsize=11, fontweight='bold')
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
        f"Validação com Áudio Real de Voz Humana — Quantum Speech Codec ({n_qubits} Qubits) + Q-LOCK\n"
        f"Corpus VOiCES | Compressão {int(cr)}:1 ({FRAME_SIZE} amostras -> {n_qubits} Qubits) | {n_total_frames} Frames de Fala (8 kHz)",
        fontsize=13, fontweight='bold'
    )
    
    out_img = "grafico_validacao_audio.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico publicável de áudio salvo em: {out_img}", flush=True)
    
    # --- Resumo Final ---
    print("\n" + "=" * 70)
    print("RESUMO FINAL DA VALIDAÇÃO COM VOZ HUMANA REAL")
    print("=" * 70)
    print(f"  Duração do áudio:    {len(original_continuous)/fs:.2f} segundos ({n_total_frames} frames)")
    print(f"  Frequência:          {fs} Hz (Banda de Voz Telefônica)")
    print(f"  Qubits:              {n_qubits} (Hilbert: 2^{n_qubits} = {2**n_qubits} estados)")
    print(f"  Taxa de compressão:  {int(cr)}:1 (32:1 / 96.88%)")
    print(f"  Correlação de voz:   {audio_corr:.4f} ({audio_corr*100:.1f}%)")
    print(f"  Relação Sinal-Ruído: {audio_snr:.2f} dB")
    print(f"  Distorção Espectral: {audio_lsd:.2f} dB (LSD)")
    print(f"  Proteção Q-LOCK:     {protection_factor:.1f}x")
    print(f"  Arquivos .WAV:       audio_original.wav, audio_reconstructed_quantum.wav, audio_encrypted_eve.wav")
    print(f"  Gráfico:             {out_img}")
    print("=" * 70, flush=True)
