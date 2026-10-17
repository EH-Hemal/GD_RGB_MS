"""Collect existing GD/WeedyRice run assets; never trains or loads checkpoints.

Run as one notebook cell or `python tools/collect_existing_run_assets.py`.
Edit EXTRA_ROOTS to point at your saved result folders or result ZIPs.
The default search is restricted to known research output folders.
"""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile

EXTRA_ROOTS = []  # e.g. [Path('/kaggle/input/my-saved-gd-results')]
INCLUDE_WEIGHTS = False  # True creates a potentially large handoff ZIP.
MAX_METADATA_MB = 32
OUT_PARENT = (Path('/kaggle/working') if Path('/kaggle/working').exists()
              else Path('/content') if Path('/content').exists() else Path.cwd())
OUT_DIR = OUT_PARENT / 'gd_release_asset_handoff'
META_EXT = {'.json', '.csv', '.txt', '.log', '.yaml', '.yml', '.py', '.ipynb'}
WEIGHT_EXT = {'.pt', '.pth', '.ckpt', '.safetensors'}
PREFIXES = ('gd_', 'weedy_', 'spatial_24m', 'spatial24m', 'ndre_', 'GD_multiseed')
SECRET_WORDS = ('credential', 'token', 'secret', 'password', 'kaggle.json', '.env')

def allowed_name(name):
    return not any(w in name.lower() for w in SECRET_WORDS)

def defaults():
    roots=[]
    for parent in [Path('/kaggle/working'), Path('/content'), Path('/content/drive/MyDrive')]:
        if parent.is_dir():
            roots.extend(p for p in parent.iterdir()
                         if p.name.startswith(PREFIXES) and p.name != OUT_DIR.name)
    for p in [Path('/kaggle/input/datasets/hemal7/gd-rgbms-db'),
              Path('/kaggle/input/hemal7/gd-rgbms-db'),
              Path('/kaggle/input/datasets/hemal7/weedy-rgb-ms'),
              Path('/content/GD-RGBMS-DB')]:
        if p.exists():roots.append(p)
    return roots

def collect(roots, out_dir, include_weights=False):
    """Copy metadata and optionally weights. Return manifest; no file execution."""
    out_dir=Path(out_dir).resolve()
    out_dir.mkdir(parents=True,exist_ok=True)
    # Never merge a new export silently with files from an earlier session.
    if (out_dir/'collection_manifest.json').exists():
        raise FileExistsError(f'Existing handoff: {out_dir}; select a fresh OUT_DIR.')
    records=[];seen=set();limit=MAX_METADATA_MB*1024**2
    def record_bytes(data,dest,origin):
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        records.append({'source':origin,'stored':str(dest.relative_to(out_dir)),
                        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'copied'})
    for idx,root in enumerate(roots):
        root=Path(root).resolve()
        if root in seen:continue
        seen.add(root)
        if not root.exists():
            records.append({'source':str(root),'status':'missing'});continue
        candidates=[root] if root.is_file() else sorted(root.rglob('*'))
        for p in candidates:
            if not p.is_file() or p.is_symlink() or out_dir in p.parents:continue
            if not allowed_name(p.name):continue
            rel=Path(root.name)/p.relative_to(root) if root.is_dir() else Path(root.name)
            dest=out_dir/'assets'/f'root_{idx:02}'/rel
            suffix=p.suffix.lower();size=p.stat().st_size
            if suffix in WEIGHT_EXT:
                if include_weights:
                    import shutil
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
                    h=hashlib.sha256()
                    with p.open('rb') as f:
                        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                    records.append({'source':str(p),'stored':str(dest.relative_to(out_dir)),
                                    'bytes':size,'sha256':h.hexdigest(),'status':'copied_weight'})
                else:records.append({'source':str(p),'bytes':size,'status':'weight_present_not_copied'})
            elif suffix in META_EXT:
                if size<=limit:record_bytes(p.read_bytes(),dest,str(p))
                else:records.append({'source':str(p),'bytes':size,'status':'metadata_over_size_limit'})
            elif suffix=='.zip':
                try:
                    with zipfile.ZipFile(p) as z:
                        for zi in z.infolist():
                            member=PurePosixPath(zi.filename)
                            if zi.is_dir() or member.is_absolute() or '..' in member.parts or '\\' in zi.filename:continue
                            if not allowed_name(member.name):continue
                            msuffix=member.suffix.lower();origin=f'{p}::{zi.filename}'
                            if msuffix in WEIGHT_EXT:
                                if include_weights:
                                    target=dest.with_suffix('')/Path(*member.parts)
                                    target.parent.mkdir(parents=True,exist_ok=True)
                                    h=hashlib.sha256()
                                    with z.open(zi) as src,target.open('wb') as dst:
                                        while b:=src.read(1024*1024):dst.write(b);h.update(b)
                                    records.append({'source':origin,'stored':str(target.relative_to(out_dir)),
                                                    'bytes':zi.file_size,'sha256':h.hexdigest(),'status':'copied_weight'})
                                else:records.append({'source':origin,'bytes':zi.file_size,'status':'weight_present_not_copied'})
                            elif msuffix in META_EXT and zi.file_size<=limit:
                                record_bytes(z.read(zi),dest.with_suffix('')/Path(*member.parts),origin)
                except (zipfile.BadZipFile,RuntimeError,OSError) as exc:
                    records.append({'source':str(p),'status':'archive_error','error':str(exc)})
    (out_dir/'collection_manifest.json').write_text(json.dumps(records,indent=2))
    return records

def save_environment(out_dir):
    packages=sorted((d.metadata.get('Name','unknown'),d.version) for d in importlib.metadata.distributions())
    env={'python':sys.version,'platform':platform.platform(),
         'note':'Environment at collection time; not automatically the original training environment.',
         'packages':dict(packages)}
    for label,cmd in [('gpu',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),
                      ('cmx_commit',['git','-C',str(OUT_PARENT/'RGBX_Semantic_Segmentation_official'),'rev-parse','HEAD'])]:
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            env[label]=p.stdout.strip() if p.returncode==0 else 'not available'
        except (OSError,subprocess.TimeoutExpired):env[label]='not available'
    (out_dir/'environment_at_collection.json').write_text(json.dumps(env,indent=2))

def main():
    roots=defaults()+[Path(x) for x in EXTRA_ROOTS]
    records=collect(roots,OUT_DIR,INCLUDE_WEIGHTS);save_environment(OUT_DIR)
    archive=OUT_DIR.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in sorted(OUT_DIR.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT_DIR))
    n=sum(r['status'].startswith('copied') for r in records)
    print(f'Collected {n} files; inventory contains {len(records)} entries.')
    print('Download:',archive)
    if n==0:print('No matching saved files found. Set EXTRA_ROOTS to your downloaded result ZIPs/folders. No experiment was rerun.')
    try:
        from IPython.display import FileLink,display
        display(FileLink(str(archive)))
    except ImportError:pass

if __name__=='__main__':main()
