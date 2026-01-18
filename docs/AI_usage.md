**AI prompt link**
https://chatgpt.com/share/696d1f61-a1c8-8008-b029-d169a4449947


**VERIFICATION**
1. **Unit Testing:** Ran python src/model.py → verified:
   - Output shape: [B, 4, 4]
   - Trace: tensor([1.0000, 1.0000, ...])
   - Hermitian: torch.allclose(rho, rho.mH) = True
2. **Training Validation:** Ran python src/train.py → achieved:
   - **Fidelity: 0.98+**
   - **Trace Distance: <0.02**
3. **Shape Debugging:** Manually fixed tensor dimension mismatches:
   - Complex shadows [B,1024,16] → real embeddings [B,1024,32]
   - Cholesky vecL [B,10] → lower triangular L [B,4,4]
4. **Constraint Checking:** Added print statements to verify:
   ```python
   print(f"Trace: {torch.diagonal(rho_pred).sum(-1)}")  
   print(f"Hermitian: {torch.allclose(rho_pred, rho_pred.mH)}")  