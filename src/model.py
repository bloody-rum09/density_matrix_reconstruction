import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer-based model for density matrix recovery
# from classical-shadow measurements

class ShadowTransformer(nn.Module):
    def __init__(self, nqubits=2, nshots=1024, d_model=64, nhead=4, nlayers=3):
        super().__init__()

        # Hilbert-space dimension
        self.dim = 2**nqubits  
        self.nshots = nshots
        # Number of independent parameters in a lower-triangular
        # Cholesky factor (real-valued parametrization)
        self.ncholesky = self.dim * (self.dim + 1) // 2  
        
        # Embed flattened (real + imag) shadow snapshots into model space
        self.shadow_embed = nn.Linear(self.dim*self.dim * 2, d_model)  

        # Standard Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # Map pooled transformer features to Cholesky parameter
        self.fc_cholesky = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model * 2, self.ncholesky), nn.Tanh()
        )
    
    def forward(self, shadows):
        # Convert complex inputs to real-valued representation
        shadows_real = torch.cat([shadows.real, shadows.imag], dim=-1)
        # Per-shot embedding
        x = self.shadow_embed(shadows_real)
        x = self.transformer(x).mean(dim=1)  
        
        # Predict Cholesky-vector parameters
        vecL = self.fc_cholesky(x)  
        # Construct lower-triangular Cholesky factor
        L = self.vec_to_lower_tri(vecL)  
        # Reconstruct valid density matrix
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

# Simple fidelity proxy loss (real-part MSE)
def fidelity_loss(rho_pred, rho_true):
    return F.mse_loss(rho_pred.real, rho_true.real)

# Sanity check on forward pass and constraints
if __name__ == "__main__":
    data = torch.load('outputs/data/dataset.pt')
    
    model = ShadowTransformer(nqubits=2, nshots=1024)
    shadows = data['shadows'][:4]
    rho_true = data['rhos_true'][:4]
    
    rho_pred = model(shadows)
    
    print(f"Input:     {shadows.shape}")
    print(f"Target:    {rho_true.shape}")
    print(f"Predicted: {rho_pred.shape}") 
    print(f"Trace:     {torch.diagonal(rho_pred, dim1=-2, dim2=-1).sum(-1)}")
    print(f"Hermitian: {torch.allclose(rho_pred, torch.conj(rho_pred.mH), atol=1e-4)}")
    print(f"MSE Loss:  {fidelity_loss(rho_pred, rho_true):.4f}")
