import numpy as np
import torch
from pathlib import Path
import argparse

# generating random n-qubit density matrix
def generate_random_rho(nqubits=2):
    dim = 2**nqubits
    # random complex state vector
    psi = np.random.normal(0, 1, dim) + 1j * np.random.normal(0, 1, dim)
    psi = psi / np.linalg.norm(psi)
    # Pure-state density matrix
    rho = np.outer(psi, np.conj(psi))
    
    #  Apply depolarizing noise to obtain a mixed state
    p_depol = np.random.uniform(0, 0.3)
    rho = (1 - p_depol) * rho + p_depol * np.eye(dim) / dim
    return rho

# generating classical shadows from a given density matrix
def shadows_from_rho(rho, n_shots=1024, nqubits=2):
    dim = 2**nqubits
    shadows = []
    
    # Single-qubit Pauli operators
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    paulibasis = [X, Y, Z]
    
    for _ in range(n_shots):
        # Randomly choose which qubit and which Pauli to measure
        qubit_idx = np.random.randint(nqubits)
        pauli_idx = np.random.randint(3)
        P_single = paulibasis[pauli_idx]
        
        # Build the full n-qubit Pauli operator
        P_full = np.eye(dim)
        slice_start = qubit_idx * 2
        slice_end = slice_start + 2
        P_full = np.eye(dim).reshape([2]*nqubits + [2]*nqubits)
        P_full[slice(slice_start, slice_end, 1), slice(slice_start, slice_end, 1)] = P_single
        
        P_full = P_full.reshape(dim, dim)
        
        # Compute expectation value Tr(P rho)
        expval = np.real(np.trace(P_full @ rho))
        # Classical-shadow snapshot estimator
        shadow_snap = (3 * expval * P_full + np.eye(dim)) / 2
        # Store flattened snapshot for downstream learning
        shadows.append(shadow_snap.flatten())
    
    return np.array(shadows)

# Dataset generation entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nqubits', type=int, default=2)
    parser.add_argument('--nshots', type=int, default=1024)
    parser.add_argument('--nsamples', type=int, default=100)
    parser.add_argument('--outdir', default='outputs/data')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.nsamples} samples...")
    rhos_true = []
    shadows_list = []
    
    for i in range(args.nsamples):
        rho = generate_random_rho(args.nqubits)
        shadows = shadows_from_rho(rho, args.nshots, args.nqubits)
        rhos_true.append(rho)
        shadows_list.append(shadows)
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{args.nsamples}")
    
    # Package dataset for PyTorch workflows
    dataset = {
        'rhos_true': torch.tensor(np.array(rhos_true), dtype=torch.complex64),
        'shadows': torch.tensor(np.array(shadows_list), dtype=torch.complex64),
        'nqubits': args.nqubits
    }
    
    torch.save(dataset, f'{args.outdir}/dataset.pt')
    print(f"SAVED: {args.outdir}/dataset.pt")

# Quick sanity checks on saved data
import torch
data = torch.load('outputs/data/dataset.pt')
# Inspect one example density matrix
print("Sample ρ_true:", torch.round(torch.real(data['rhos_true'][0]), decimals=2))
# Hermiticity check
print("Hermitian->", torch.allclose(data['rhos_true'][0], torch.conj(data['rhos_true'][0].T)))
# Trace normalization check
print("Trace=1->", torch.abs(torch.trace(data['rhos_true'][0]) - 1) < 1e-5)

