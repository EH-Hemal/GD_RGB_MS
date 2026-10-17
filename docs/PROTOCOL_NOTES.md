# Protocol notes

- GD supplied standard split: 280/60/60. The notebooks load `train_list.txt`, `val_list.txt`, `test_list.txt` from the dataset; the files themselves were not embedded in the notebooks.
- Input `(width,height)=(896,672)`. RGB plus the stored three-channel MS composite (Green, Red Edge, NIR). Original source uses PIL RGB conversion, bilinear image resize, nearest mask resize, and hard-coded mask decoding; keep these unchanged when reproducing. Verify actual band order with the dataset metadata.
- Main proposed architecture: 7,950,483 total parameters including 450 training-only auxiliary-head parameters; 7,950,033 deployment parameters. Shared features feed back into both branches. Auxiliary heads act on the deepest branch features; these are not wholly isolated after fusion feedback.
- Main controlled settings typically AdamW learning rate/weight decay 1e-4, maximum70 epochs, early stopping12, LR plateau patience4/factor0.5, physical batch2/accumulation2. Use each original source as authority; do not impose these settings on all experiments.
- Strong baselines: 70 epochs for42, 50 for123/2026; effective batch4 with per-model physical-batch differences. Original code downloads CMX from an unpinned repository. Record the historical commit if recoverable.
- Standard final fine threshold grid in these sources is0.20–0.80 by0.01; the buffered runner uses0.20–0.95. Do not replace these distinct historical settings with a single manuscript assertion without checking actual configs.
- Buffered split retains207 train/40val/41test and excludes112. Exact membership must come from the run artifacts. Three seeds share the split. This is not spatially blocked5-fold CV.
- Cross acquisition uses source-only development train/validation, then fresh training on ALL source images for the selected duration, and final unseen-target evaluation. A217/B183; development sizes174/43 and146/37. Earlier DBSCAN+NumPy code differs from the KMeans+sklearn recheck/remaining-runs code. Use the actual per-run lists.
- WeedyRice experiments retrain using official dataset lists. Unimodal budget50 vs later proposed budget70: preserve this difference.
- Saved source notebook outputs can represent earlier runs. `results/reported_summary.csv` is a transcription of author-confirmed manuscript aggregates, not a recomputation from all raw runs.
- Old image-wise statistical tests describe seed42 image variability, not between-seed training variability. Spatial dependence is not removed by a paired test or Holm correction. The original statistical code/data were not supplied here.
