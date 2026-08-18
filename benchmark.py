import time
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

# Fixando seed para reprodutibilidade estrita
torch.manual_seed(42)

# =====================================================================
# 1. DEFINIÇÃO DO CIRCUITO QUÂNTICO E MODELO HÍBRIDO
# =====================================================================
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

def build_encoder():
    return nn.Sequential(
        nn.Linear(8192, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Linear(1024, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 16),
        nn.LayerNorm(16),
        nn.GELU(),
        nn.Linear(16, 4)
    )

def build_decoder():
    return nn.Sequential(
        nn.Linear(4, 16),
        nn.LayerNorm(16),
        nn.GELU(),
        nn.Linear(16, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Linear(1024, 8192)
    )

class QuantumBottleneckAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.quantum_node = qlayer
        self.decoder = build_decoder()

    def forward(self, x):
        latent_classical = self.encoder(x)
        latent_bounded = torch.tanh(latent_classical) * torch.pi
        latent_quantum = self.quantum_node(latent_bounded)
        reconstructed = self.decoder(latent_quantum)
        return reconstructed

# =====================================================================
# 2. DEFINIÇÃO DO MODELO 100% CLÁSSICO
# =====================================================================
class ClassicalBottleneckAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        # Gargalo clássico não-linear em R^4 com regularizacao de escala
        self.classical_bottleneck = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.Tanh()
        )
        self.decoder = build_decoder()

    def forward(self, x):
        latent_classical = self.encoder(x)
        latent_processed = self.classical_bottleneck(latent_classical)
        reconstructed = self.decoder(latent_processed)
        return reconstructed

# =====================================================================
# 3. GERAÇÃO DE SINAIS ESTRUTURADOS
# =====================================================================
def generate_structured_signals(batch_size=4, length=8192):
    t = torch.linspace(0, 1, length)
    signals = []
    for i in range(batch_size):
        f1 = 3.0 * (i + 1)
        f2 = 7.0 * (i + 1)
        phase = i * (torch.pi / 3)
        sig = torch.sin(2 * torch.pi * f1 * t + phase) + 0.5 * torch.cos(2 * torch.pi * f2 * t)
        sig = sig / sig.abs().max()
        signals.append(sig)
    return torch.stack(signals)

# =====================================================================
# 4. EXECUÇÃO DO BENCHMARK LADO A LADO
# =====================================================================
if __name__ == "__main__":
    num_epochs = 150
    batch_size = 4
    structured_signal = generate_structured_signals(batch_size, 8192)
    criterion = nn.MSELoss()

    print("=" * 65)
    print("BENCHMARK: AUTOENCODER CLÁSSICO vs. AUTOENCODER QUÂNTICO HÍBRIDO")
    print(f"Lote: {batch_size} sinais | Dimensão: 8192 -> 4 -> 8192 | Épocas: {num_epochs}")
    print("=" * 65)

    # --- Treinamento do Modelo Clássico ---
    print("\n[1/2] Treinando Modelo Clássico...")
    torch.manual_seed(42)
    model_classical = ClassicalBottleneckAutoencoder()
    opt_classical = torch.optim.AdamW(model_classical.parameters(), lr=3e-3, weight_decay=1e-5)
    sched_classical = torch.optim.lr_scheduler.CosineAnnealingLR(opt_classical, T_max=num_epochs, eta_min=1e-5)
    
    start_time_c = time.time()
    loss_hist_c = []
    for epoch in range(num_epochs):
        opt_classical.zero_grad()
        out_c = model_classical(structured_signal)
        loss_c = criterion(out_c, structured_signal)
        loss_c.backward()
        opt_classical.step()
        sched_classical.step()
        loss_hist_c.append(loss_c.item())
    time_classical = time.time() - start_time_c
    print(f" -> Concluído em {time_classical:.2f}s | MSE Final: {loss_hist_c[-1]:.8f}")

    # --- Treinamento do Modelo Quântico ---
    print("\n[2/2] Treinando Modelo Quântico...")
    torch.manual_seed(42)
    model_quantum = QuantumBottleneckAutoencoder()
    opt_quantum = torch.optim.AdamW(model_quantum.parameters(), lr=3e-3, weight_decay=1e-5)
    sched_quantum = torch.optim.lr_scheduler.CosineAnnealingLR(opt_quantum, T_max=num_epochs, eta_min=1e-5)
    
    start_time_q = time.time()
    loss_hist_q = []
    for epoch in range(num_epochs):
        opt_quantum.zero_grad()
        out_q = model_quantum(structured_signal)
        loss_q = criterion(out_q, structured_signal)
        loss_q.backward()
        opt_quantum.step()
        sched_quantum.step()
        loss_hist_q.append(loss_q.item())
    time_quantum = time.time() - start_time_q
    print(f" -> Concluído em {time_quantum:.2f}s | MSE Final: {loss_hist_q[-1]:.8f}")

    # --- Teste de Robustez com Ruído Adicionado (Generalização) ---
    print("\n[3/3] Executando Teste de Robustez com Ruído Gaussiano...")
    noise_level = 0.20
    torch.manual_seed(123)
    noisy_signal = structured_signal + noise_level * torch.randn_like(structured_signal)
    
    with torch.no_grad():
        out_noisy_c = model_classical(noisy_signal)
        out_noisy_q = model_quantum(noisy_signal)
        
        # Erro de reconstrução em relação ao sinal limpo original
        denoise_mse_c = criterion(out_noisy_c, structured_signal).item()
        denoise_mse_q = criterion(out_noisy_q, structured_signal).item()

    print(f" -> MSE Clássico sob Ruído: {denoise_mse_c:.6f}")
    print(f" -> MSE Quântico sob Ruído: {denoise_mse_q:.6f}")

    # =====================================================================
    # 5. GERAÇÃO DO GRÁFICO COMPARATIVO COMPLETO DO BENCHMARK
    # =====================================================================
    print("\nGerando grafico_benchmark.png...")
    orig = structured_signal.detach().cpu().numpy()
    rec_c = out_c.detach().cpu().numpy()
    rec_q = out_q.detach().cpu().numpy()
    
    rec_noisy_c = out_noisy_c.detach().cpu().numpy()
    rec_noisy_q = out_noisy_q.detach().cpu().numpy()
    noisy_np = noisy_signal.detach().cpu().numpy()

    pts = 600
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Subplot 1: Curva de Convergência (Log Scale)
    axs[0, 0].semilogy(range(1, num_epochs + 1), loss_hist_c, label=f"Clássico (Final: {loss_hist_c[-1]:.1e})", color='#1f77b4', lw=2)
    axs[0, 0].semilogy(range(1, num_epochs + 1), loss_hist_q, label=f"Quântico (Final: {loss_hist_q[-1]:.1e})", color='#d62728', lw=2, linestyle='--')
    axs[0, 0].set_title("1. Velocidade de Convergência (Escala Log)", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Épocas")
    axs[0, 0].set_ylabel("MSE Loss (Log)")
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].grid(True, which="both", linestyle='--', alpha=0.5)

    # Subplot 2: Comparação de Reconstrução do Sinal Limpo
    axs[0, 1].plot(orig[0, :pts], label="Original Limpo", color='black', lw=2.5, alpha=0.8)
    axs[0, 1].plot(rec_c[0, :pts], label="Reconstrução Clássica", color='#1f77b4', lw=1.8, linestyle=':')
    axs[0, 1].plot(rec_q[0, :pts], label="Reconstrução Quântica", color='#d62728', lw=1.8, linestyle='--')
    axs[0, 1].set_title(f"2. Sinal Limpo: Comparação (Primeiros {pts} pts)", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Tempo (t)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Teste de Desruidificação / Robustez
    axs[1, 0].plot(noisy_np[0, :pts], label="Entrada com Ruído (+20%)", color='gray', alpha=0.5, lw=1)
    axs[1, 0].plot(orig[0, :pts], label="Alvo Limpo Original", color='black', lw=2)
    axs[1, 0].plot(rec_noisy_c[0, :pts], label=f"Filtro Clássico (MSE: {denoise_mse_c:.4f})", color='#1f77b4', lw=1.8)
    axs[1, 0].plot(rec_noisy_q[0, :pts], label=f"Filtro Quântico (MSE: {denoise_mse_q:.4f})", color='#d62728', lw=1.8, linestyle='--')
    axs[1, 0].set_title("3. Teste de Robustez e Filtragem de Ruído", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Tempo (t)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 4: Gráfico de Barras de Métricas Resumo
    metricas = ['MSE Treino (x10^-4)', 'MSE com Ruído', 'Tempo Treino (s)']
    valores_c = [loss_hist_c[-1] * 10000, denoise_mse_c, time_classical]
    valores_q = [loss_hist_q[-1] * 10000, denoise_mse_q, time_quantum]
    
    x = range(len(metricas))
    width = 0.35
    rects1 = axs[1, 1].bar([i - width/2 for i in x], valores_c, width, label='Clássico', color='#1f77b4')
    rects2 = axs[1, 1].bar([i + width/2 for i in x], valores_q, width, label='Quântico Híbrido', color='#d62728')
    axs[1, 1].set_title("4. Comparativo Direto de Desempenho", fontsize=12, fontweight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(metricas, fontsize=10)
    axs[1, 1].legend()
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("BENCHMARK: Autoencoder Clássico vs. Autoencoder Quântico (8192 -> 4 -> 8192)", fontsize=14, fontweight='bold')
    output_benchmark_img = "grafico_benchmark.png"
    plt.savefig(output_benchmark_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de benchmark salvo com sucesso em: {output_benchmark_img}")
