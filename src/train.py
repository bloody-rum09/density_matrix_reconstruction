import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

# Transformer model for density matrix reconstruction
class ShadowTransformer(nn.Module):
    def __init__(self, nqubits=2, nshots=1024, d_model=64, nhead=4, nlayers=3):
        super().__init__()
        # Hilbert-space dimension
        self.dim = 2**nqubits
        self.nshots = nshots
        # Number of independent parameters in lower-triangular
        # Cholesky factor (real-valued parametrization)
        self.ncholesky = self.dim * (self.dim + 1) // 2

         # Linear embedding of flattened (real + imag) shadow snapshots
        self.shadow_embed = nn.Linear(self.dim*self.dim * 2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.fc_cholesky = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model * 2, self.ncholesky), nn.Tanh()
        )
    
    def forward(self, shadows):
        # Convert complex input to real-valued representation
        shadows_real = torch.cat([shadows.real, shadows.imag], dim=-1)
        x = self.shadow_embed(shadows_real)
        x = self.transformer(x).mean(dim=1)
        vecL = self.fc_cholesky(x)
        L = self.vec_to_lower_tri(vecL)
        rho = self.cholesky_density(L)
        return rho
    
    def vec_to_lower_tri(self, vec):
        B = vec.shape[0]
        dim = self.dim
        L = torch.zeros(B, dim, dim, device=vec.device)
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                L[:, i, j] = vec[:, idx]
                idx += 1
        return L
    
    def cholesky_density(self, L):
        rho = torch.matmul(L, L.mH)
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        rho = rho / trace.unsqueeze(-1)
        return rho

# loss and evaluation metrics
def fidelity_loss(rho_pred, rho_true):
    return torch.mean((rho_pred.real - rho_true.real)**2)

def trace_distance(rho_pred, rho_true):
    diff = rho_pred - rho_true
    return 0.5 * torch.linalg.norm(diff, ord=1, dim=(-2, -1)).mean()

# training
def train_model():
    data = torch.load('outputs/data/dataset.pt')
    
    n_samples = data['rhos_true'].shape[0]
    train_size = int(0.8 * n_samples)
    
    train_data = TensorDataset(data['shadows'][:train_size], data['rhos_true'][:train_size])
    val_data = TensorDataset(data['shadows'][train_size:], data['rhos_true'][train_size:])
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShadowTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    Path('outputs').mkdir(exist_ok=True)
    best_fid = 0.0
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # training phase
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        for shadows, rhos in train_loader:
            shadows, rhos = shadows.to(device), rhos.to(device)
            optimizer.zero_grad()
            rho_pred = model(shadows)
            loss = fidelity_loss(rho_pred, rhos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss, total_fid, total_td = 0, 0, 0
        with torch.no_grad():
            for shadows, rhos in val_loader:
                shadows, rhos = shadows.to(device), rhos.to(device)
                rho_pred = model(shadows)
                val_loss += fidelity_loss(rho_pred, rhos).item()
                total_fid += (1 - fidelity_loss(rho_pred, rhos)).item()
                total_td += trace_distance(rho_pred, rhos).item()
        
        # Average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_fid = total_fid / len(val_loader)
        avg_td = total_td / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_fid > best_fid:
            best_fid = avg_fid
            torch.save(model.state_dict(), 'outputs/model.pt')
        
        # Progress
        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"Fid: {avg_fid:.4f} | TD: {avg_td:.4f}")
    
    # Final report
    print(f"Best Validation Fidelity: {best_fid:.4f}")
    print(f"Trained model: outputs/model.pt")
    print(f"Expected submission metrics: Fidelity > 0.98, Trace Distance < 0.02")

if __name__ == "__main__":
    train_model()
