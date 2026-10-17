"""Optional CPU/CUDA smoke check after installing PyTorch; no training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.proposed import AuxiliarySupervisedGatedUNetNoNDRE

model=AuxiliarySupervisedGatedUNetNoNDRE().eval()
total=sum(p.numel() for p in model.parameters())
aux=sum(p.numel() for n,p in model.named_parameters() if 'auxiliary_head' in n)
assert total==7_950_483,(total,aux)
assert total-aux==7_950_033,(total,aux)
with torch.inference_mode():
    x=torch.zeros(1,6,64,96)
    y,g,r,s=model(x,return_gates=True,return_aux=True)
    assert y.shape==r.shape==s.shape==(1,1,64,96)
    for gate in g:assert torch.allclose(gate.sum(1),torch.ones_like(gate[:,0]),atol=1e-6)
print('PASS: parameters, auxiliary outputs, output size and gate normalization.')
