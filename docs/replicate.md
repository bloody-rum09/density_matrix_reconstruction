# Replication Guide: Density Matrix Reconstruction

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/bloody-rum/density-matrix-reconstruction.git
cd density_matrix_reconstruction

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)  
source venv/bin/activate

# Install dependencies
pip install torch numpy qutip matplotlib

python src/data_gen.py

python src/train.py

Epoch  10 | Loss: 0.0123 | Fidelity: 0.89 | Trace Distance: 0.045
Epoch  50 | Loss: 0.0021 | Fidelity: 0.96 | Trace Distance: 0.023  
Epoch  99 | Loss: 0.0011 | Fidelity: **0.98+** | Trace Distance: **<0.02**
python src/model.py

python src/model.py
