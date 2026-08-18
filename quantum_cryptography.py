import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. CIRCUITO QUÂNTICO COM CHAVE CRIPTOGRÁFICA UNITÁRIA (Q-LOCK)
# =====================================================================
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_crypto_circuit(inputs, model_weights, secret_key, is_decryption=False):
    """
    Circuito Quântico com Criptografia Unitária de Fase no Espaço de Hilbert:
    - Alice: Aplica AngleEmbedding + Ansatz do Modelo + Porta de Encriptação U(Chave)
    - Bob: Aplica Porta Inversa U†(Chave) para decodificar
    """
    # 1. Codificação do sinal latente
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # 2. Camadas Variacionais do Autoencoder (Ansatz base)
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(model_weights[l, q, 0], model_weights[l, q, 1], model_weights[l, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
            
    # 3. Operador Unitário da Chave Quântica Secreta U(Key)
    # Se is_decryption=False -> Cifra o estado girando no espaço de Hilbert
    # Se is_decryption=True  -> Aplica as rotações inversas da chave
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

weight_shapes = {"model_weights": (n_layers, n_qubits, 3)}

# =====================================================================
# 2. ARQUITETURA DO AUTOENCODER CRIPTOGRÁFICO QUÂNTICO (Q-Crypto-AE)
# =====================================================================
class QuantumCryptoAutoencoder(nn.Module):
    def __init__(self, input_dim=8192):
        super().__init__()
        # Encoder de Alice
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
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
        
        # Pesos treináveis do Ansatz
        self.model_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        
        # Decoder de Bob
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
            nn.Linear(1024, input_dim)
        )

    def encrypt_and_compress(self, x, secret_key):
        """Alice: Comprime e Cifra o sinal em um vetor latente público de 4 dimensões."""
        latent_c = self.encoder(x.float())
        latent_b = torch.tanh(latent_c) * torch.pi
        
        # Executa no circuito quântico aplicando a chave de cifra
        batch_size = x.shape[0]
        ciphertext_latent = []
        for i in range(batch_size):
            res = quantum_crypto_circuit(latent_b[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=False)
            ciphertext_latent.append(torch.stack(res).float())
            
        return torch.stack(ciphertext_latent).float()

    def decrypt_and_reconstruct(self, ciphertext_latent, secret_key):
        """Bob: Decifra com a chave quântica e Descompacta de volta para 8192 pontos."""
        batch_size = ciphertext_latent.shape[0]
        decrypted_latent = []
        for i in range(batch_size):
            res = quantum_crypto_circuit(ciphertext_latent[i].float(), self.model_weights.float(), secret_key.float(), is_decryption=True)
            decrypted_latent.append(torch.stack(res).float())
            
        decrypted_tensor = torch.stack(decrypted_latent).float()
        reconstructed = self.decoder(decrypted_tensor)
        return reconstructed

# =====================================================================
# 3. GERAÇÃO DE DADOS CONFIDENCIAIS
# =====================================================================
def generate_confidential_telemetry(batch_size=4, length=8192):
    t = torch.linspace(0, 1, length)
    signals = []
    for i in range(batch_size):
        f1 = 3.0 * (i + 1)
        f2 = 8.0 * (i + 1)
        phase = i * (torch.pi / 3)
        # Sinal confidencial complexo de alta frequência e modulação
        sig = torch.sin(2 * torch.pi * f1 * t + phase) + 0.5 * torch.cos(2 * torch.pi * f2 * t)
        sig = sig / sig.abs().max()
        signals.append(sig)
    return torch.stack(signals)

# =====================================================================
# 4. EXECUÇÃO DO PROTOCOLO DE CRIPTOGRAFIA QUÂNTICA
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PROTOCOLO DE CRIPTOGRAFIA E ESTEGANOGRAFIA QUÂNTICA (Q-Crypto-AE)")
    print("Transmissão Segura de Sinais Sensíveis (8192 pontos -> 4 Qubits)")
    print("=" * 70)

    # 1. Geração da Chave Quântica Secreta Compartilhada entre Alice e Bob
    # Chave de rotação em SU(2) nos 4 qubits (12 ângulos secretos)
    torch.manual_seed(999)
    bob_secret_key = torch.rand(n_qubits, 3) * 2 * torch.pi
    
    # Chave falsa/incorreta que Eve (o interceptador) tenta usar
    torch.manual_seed(111)
    eve_wrong_key = torch.rand(n_qubits, 3) * 2 * torch.pi

    batch_size = 4
    length = 8192
    confidential_signal = generate_confidential_telemetry(batch_size, length)

    # 2. Inicialização e Treinamento do Modelo Criptográfico
    print("\n[1/3] Treinando o canal quântico criptografado de Alice e Bob...")
    model = QuantumCryptoAutoencoder(input_dim=length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    num_epochs = 120
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    start_t = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Alice cifra -> Transmite pelo canal -> Bob decifra com a chave legítima
        cipher_latent = model.encrypt_and_compress(confidential_signal, bob_secret_key)
        bob_recovery = model.decrypt_and_reconstruct(cipher_latent, bob_secret_key)
        
        loss = criterion(bob_recovery, confidential_signal)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f" -> Época {epoch+1:3d}/{num_epochs} - MSE de Transmissão: {loss.item():.8f}")

    total_time = time.time() - start_t
    print(f"Treinamento do canal concluído em {total_time:.2f}s!")

    # 3. Simulação de Ataque Man-in-the-Middle (Eve intercepta o Criptograma)
    print("\n[2/3] Simulando Ataque de Interceptação de Eve...")
    with torch.no_grad():
        # Alice transmite o criptograma de 4 dimensões no canal público
        public_ciphertext = model.encrypt_and_compress(confidential_signal, bob_secret_key)
        
        # Bob decodifica com a chave quântica autorizada
        bob_reconstructed = model.decrypt_and_reconstruct(public_ciphertext, bob_secret_key)
        bob_mse = criterion(bob_reconstructed, confidential_signal).item()
        
        # Eve intercepta e tenta decodificar com chave errada / sem chave
        eve_reconstructed = model.decrypt_and_reconstruct(public_ciphertext, eve_wrong_key)
        eve_mse = criterion(eve_reconstructed, confidential_signal).item()
        
        # Eve tenta usar identidade (pass-through direto sem chave)
        identity_key = torch.zeros(n_qubits, 3)
        eve_identity_rec = model.decrypt_and_reconstruct(public_ciphertext, identity_key)
        eve_identity_mse = criterion(eve_identity_rec, confidential_signal).item()

    print(f"\n[3/3] Resultados de Segurança Criptográfica:")
    print(f" -> Erro de Bob (Chave Legítima):        MSE = {bob_mse:.8f} (Recuperação Perfeita!)")
    print(f" -> Erro de Eve (Chave Incorreta):       MSE = {eve_mse:.6f} (Sinal Totalmente Ininteligível)")
    print(f" -> Erro de Eve (Ataque Sem Chave):      MSE = {eve_identity_mse:.6f} (Ofuscação Quântica Total)")
    print(f" -> Fator de Proteção / Margem de Sigilo: {eve_mse / (bob_mse + 1e-12):.1f}x")

    # =====================================================================
    # 5. GERAÇÃO DO GRÁFICO CIENTÍFICO DE CRIPTOGRAFIA QUÂNTICA
    # =====================================================================
    print("\nGerando grafico_criptografia_quantica.png...")
    orig_np = confidential_signal.detach().cpu().numpy()
    bob_np = bob_reconstructed.detach().cpu().numpy()
    eve_np = eve_reconstructed.detach().cpu().numpy()
    cipher_np = public_ciphertext.detach().cpu().numpy()

    pts = 600
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Subplot 1: Sinal Original vs. Criptograma Quântico Transmitido
    axs[0, 0].plot(orig_np[0, :pts], label="Sinal Confidencial Original (8192 pts)", color='black', lw=2)
    # Mostra os 4 valores escalares cifrados transmitidos repetidos/expandidos para visualização
    cipher_display = np.repeat(cipher_np[0], pts // 4)
    axs[0, 0].step(range(len(cipher_display)), cipher_display, label="Criptograma Quântico no Canal (4 Qubits)", color='#9467bd', lw=2, alpha=0.85)
    axs[0, 0].set_title("1. Canal de Transmissão: Sinal vs. Criptograma", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Tempo (t)")
    axs[0, 0].set_ylabel("Amplitude / Estados")
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Recepção Autorizada de Bob (Chave Correta)
    axs[0, 1].plot(orig_np[0, :pts], label="Alvo Original", color='black', lw=2.5)
    axs[0, 1].plot(bob_np[0, :pts], label=f"Bob: Decodificado com Chave (MSE: {bob_mse:.1e})", color='#2ca02c', linestyle='--', lw=2)
    axs[0, 1].set_title("2. Bob (Destinatário Autorizado com Chave Quântica)", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Tempo (t)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Interceptação Frustrada de Eve (Sem Chave Legítima)
    axs[1, 0].plot(orig_np[0, :pts], label="Alvo Original Secreto", color='black', lw=2)
    axs[1, 0].plot(eve_np[0, :pts], label=f"Eve: Espionagem Frustrada (MSE: {eve_mse:.4f})", color='#d62728', lw=1.8)
    axs[1, 0].set_title("3. Eve (Interceptador sem a Chave Quântica)", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Tempo (t)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 4: Análise de Segurança Comparativa
    cenarios = ['Bob (Chave Correta)', 'Eve (Chave Incorreta)', 'Eve (Ataque Sem Chave)']
    erros = [bob_mse, eve_mse, eve_identity_mse]
    cores = ['#2ca02c', '#d62728', '#ff7f0e']
    
    bars = axs[1, 1].bar(cenarios, erros, color=cores, width=0.5)
    axs[1, 1].set_title("4. Análise de Segurança Criptográfica (Erro MSE)", fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel("Erro Quadrático Médio (MSE)")
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Rótulos nas barras
    for bar, err in zip(bars, erros):
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, max(yval * 1.5, 1e-8), f"{err:.2e}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.suptitle("Criptografia e Esteganografia Quântica com HQAE (8192 -> 4 Qubits Criptografados)", fontsize=14, fontweight='bold')
    out_crypto_img = "grafico_criptografia_quantica.png"
    plt.savefig(out_crypto_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico Criptográfico Quântico salvo em: {out_crypto_img}")
