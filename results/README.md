# Result provenance

`reported_summary.csv` preserves selected final manuscript aggregates with a source column. It is not a synthetic set of measurements and must not be used to infer missing raw values. `available_per_seed.csv` transcribes only explicitly supplied per-seed values; seed42 U-Net++ is a rounded historical reference.

`source_logs/` holds text outputs from the selected saved notebook cells. Some are incomplete or earlier experiments, as recorded in the experiment index. A saved source log is not automatically the final manuscript record. No output images or unrelated exploratory notebook cells were copied.

To compute sample SD from measured per-seed rows, run `python tools/summarize_runs.py results/available_per_seed.csv --out results/recomputed_available_summary.csv`. It groups by protocol/model, requires unique seeds and never fills missing metrics. Rounded input may produce slightly different last digits from full-precision original files.

Attach full-precision metrics, per-image scores, threshold scans and training histories from the original outputs before claiming complete result reproduction.
