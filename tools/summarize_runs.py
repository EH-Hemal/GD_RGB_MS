"""Summarize measured run-level CSV rows without inferring missing values."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev

def summarize(path):
    groups=defaultdict(list);seen=set()
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            key=(row['protocol'],row['model'],int(row['seed']))
            if key in seen:raise ValueError(f'Duplicate protocol/model/seed: {key}')
            seen.add(key)
            for metric in ['dice','iou','precision','recall']:
                row[metric]=float(row[metric])
                if not math.isfinite(row[metric]) or not 0<=row[metric]<=1:raise ValueError(f'Invalid {metric} in {key}')
            groups[key[:2]].append(row)
    result=[]
    for (protocol,model),rows in sorted(groups.items()):
        out={'protocol':protocol,'model':model,'n_seeds':len(rows),'seeds':';'.join(str(x) for x in sorted(int(r['seed']) for r in rows))}
        for k in ['dice','iou','precision','recall']:
            vals=[r[k] for r in rows];out[k+'_mean']=mean(vals);out[k+'_sample_sd']=stdev(vals) if len(vals)>1 else ''
        result.append(out)
    if not result:raise ValueError('No measured rows.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--out',required=True);a=p.parse_args()
    rows=summarize(a.csv);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(out)
