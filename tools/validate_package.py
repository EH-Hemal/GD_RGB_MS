"""Static packaging checks. Does not execute training or claim CUDA validation."""
import ast
import hashlib
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    checks=[]
    manifest=json.loads((root/'docs/source_manifest.json').read_text())
    for m in manifest:
        nb=json.loads((root/m['notebook']).read_text())
        cells=[c for c in nb['cells'] if c['cell_type']=='code']
        assert len(cells)==1,m['notebook']
        code=cells[0]['source'];code=''.join(code) if isinstance(code,list) else code
        compile(code,m['notebook'],'exec')
        assert hashlib.sha256(code.encode()).hexdigest()==m['code_sha256'],m['notebook']
        assert not cells[0]['outputs']
        checks.append(m['notebook'])
    for p in root.rglob('*.py'):compile(p.read_text(),str(p),'exec')
    for p in root.rglob('*.json'):json.loads(p.read_text())
    print(f'PASS: {len(checks)} experiment notebooks match manifest; Python syntax and JSON checked.')
    print('Not tested: training, CUDA forward passes, historical split membership or checkpoint loading.')
    print('Missing run assets remain listed in docs/RELEASE_STATUS.md.')

if __name__=='__main__':main()
