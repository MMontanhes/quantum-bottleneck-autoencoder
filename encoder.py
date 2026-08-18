import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

# 1. Miolo quantico (4 qubits com circuito variacional e entrelacamento CNOT)
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Codificacao angular dos 4 valores latentes em rotacao Y nos 4 qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # Camadas variacionais parametrizadas com portas CNOT em anel
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
            
    # Medicao dos valores esperados de Pauli-Z em todos os 4 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# 2. Arquitetura Hibrida Estabilizada (8192 -> 4 Qubits -> 8192)
class QuantumBottleneckAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder estruturado com LayerNorm e GELU para fluxo otimo de gradiente
        self.encoder = nn.Sequential(
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
        
        # Interface quantica
        self.quantum_node = qlayer
        
        # Decoder estruturado espelhado
        self.decoder = nn.Sequential(
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

    def forward(self, x):
        # Comprime ate 4 dimensoes
        latent_classical = self.encoder(x)
        
        # Mapeia estritamente para o espaco angular [-pi, pi]
        latent_bounded = torch.tanh(latent_classical) * torch.pi
        
        # Processamento no espaco de Hilbert com 4 qubits entrelacados
        latent_quantum = self.quantum_node(latent_bounded)
        
        # Reconstrói para 8192 dimensoes
        reconstructed = self.decoder(latent_quantum)
        return reconstructed


def generate_structured_signals(batch_size=4, length=8192):
    """Gera sinais estruturados compostos por diferentes harmonicos e fases."""
    t = torch.linspace(0, 1, length)
    signals = []
    for i in range(batch_size):
        f1 = 3.0 * (i + 1)
        f2 = 7.0 * (i + 1)
        phase = i * (torch.pi / 3)
        # composicao harmonica estruturada
        sig = torch.sin(2 * torch.pi * f1 * t + phase) + 0.5 * torch.cos(2 * torch.pi * f2 * t)
        # normalizacao
        sig = sig / sig.abs().max()
        signals.append(sig)
    return torch.stack(signals)


# 3. Teste e treino otimizado
if __name__ == "__main__":
    model = QuantumBottleneckAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    
    num_epochs = 150
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # Gerando sinal estruturado continuo de 8192 pontos
    batch_size = 4
    structured_signal = generate_structured_signals(batch_size, 8192)
    
    loss_history = []
    
    print("=" * 60)
    print(f"iniciando compressao/reconstrucao ALTA PRECISÃO")
    print(f"lote: {batch_size} sinais | dimensao: 8192 pontos | epocas: {num_epochs}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(structured_signal)
        loss = criterion(output, structured_signal)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_loss = loss.item()
        loss_history.append(current_loss)
        
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"epoca {epoch+1:3d}/{num_epochs} - perda mse: {current_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.5f}")
    
    print("=" * 60)
    print("saida final shape:", output.shape)
    print(f"perda final mse: {loss_history[-1]:.6f}")
    
    # 4. Geracao do grafico comparativo
    print("gerando grafico comparativo...")
    orig = structured_signal.detach().cpu().numpy()
    recon = output.detach().cpu().numpy()
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Subplot 1: Curva de Perda
    axs[0, 0].plot(range(1, num_epochs + 1), loss_history, color='#007acc', lw=2)
    axs[0, 0].set_title("Evolução da Perda (MSE Loss)", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Épocas")
    axs[0, 0].set_ylabel("Erro Quadrático Médio (MSE)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Amostra 1 (Zoom nos primeiros 1000 pontos)
    pts = 1000
    axs[0, 1].plot(orig[0, :pts], label="Original", color='#2ca02c', lw=2.5)
    axs[0, 1].plot(recon[0, :pts], label="Reconstruído (Quântico)", color='#d62728', linestyle='--', lw=2)
    axs[0, 1].set_title(f"Amostra 1: Comparação (Primeiros {pts} pontos)", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Amostra Temporal (t)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 3: Amostra 2 (Zoom nos primeiros 1000 pontos)
    axs[1, 0].plot(orig[1, :pts], label="Original", color='#1f77b4', lw=2.5)
    axs[1, 0].plot(recon[1, :pts], label="Reconstruído (Quântico)", color='#ff7f0e', linestyle='--', lw=2)
    axs[1, 0].set_title(f"Amostra 2: Comparação (Primeiros {pts} pontos)", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Amostra Temporal (t)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 4: Resíduo / Erro Absoluto (Amostra 1)
    residuo = abs(orig[0, :] - recon[0, :])
    axs[1, 1].plot(residuo[:pts], color='#9467bd', lw=1.5)
    axs[1, 1].set_title(f"Erro Absoluto Residual (|Original - Reconstruído|)", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("Amostra Temporal (t)")
    axs[1, 1].set_ylabel("Erro Absoluto")
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    output_img = "grafico_comparativo.png"
    plt.suptitle("Quantum Bottleneck Autoencoder (8192 -> 4 Qubits com CNOT -> 8192)", fontsize=14, fontweight='bold')
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"grafico salvo com sucesso em: {output_img}")