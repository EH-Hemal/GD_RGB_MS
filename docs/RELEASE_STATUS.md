# Assets required to complete the manuscript release

The code package has been assembled from both supplied notebooks and the subsequent supplied runners. Code availability and historical-result verification are different tasks. No new training was executed when packaging.

| Item | What was found | Still required |
|---|---|---|
| Proposed standard model | Six-channel seed-42 source and architecture | Actual seed42/123/2026 run folders, additional-seed driver/configs, best checkpoints and full-precision metrics |
| Plain U-Net | Original source and partial older output | The FINAL three-seed RGB/MS notebook and result ZIPs; saved older RGB Dice 0.7459 differs from author-final 0.7398 |
| Residual-scSE unimodal | MS seed42 source and RGB/MS123/2026 runner | Final RGB42 source/export and authoritative per-seed metrics matching the manuscript; saved MS results and supplied aggregate require reconciliation |
| Early fusion | All-three-seed source and output | Checkpoints, original metric CSVs and image-wise outputs |
| Component ablations | Seed42, later123/2026 sources, gated2026 output | Exact final per-seed exports; update the stale approximate expected-count annotation from7,932,401 to the measured7,932,769 (seed42 source already expects7,932,769) |
| U-Net++ | 123/2026 runner, author-confirmed final metrics; architecture | Original seed42 training configuration/checkpoint; all per-image exports |
| Strong baselines | All3seeds sources and outputs | Checkpoints, metrics, CMX commit SHA, training environment |
| Buffered24m | Generator, 3seed runner and completed text log | Actual train/val/test/excluded lists or protocol ZIP, checkpoints and CSVs |
| Cross acquisition | Recheck123 plus 4remaining-runs code/log | Earlier A-to-B42 final config/results, all direction/seed development+final membership files and checkpoints |
| WeedyRice | All relevant source variants and partial saved logs | Official split copies and final exports matching the author's final reported results, especially proposed42/2026 |
| NDRE confirmation | 7channel source and external-driver invocation | `run_ndre_confirmation.py`, final6/7channel configurations/results for each reported seed |
| Statistics | Manuscript descriptions only; no Wilcoxon implementation found in either notebook | Original analysis code, same-image seed42 score CSVs, bootstrap settings and any revised spatial analysis |
| Coverage and gate analysis | Source code and selected output | Per-image coverage and gate CSVs, selected threshold, model checkpoints |

Do not reconstruct precision/recall from Dice or fabricate missing per-seed values. Do not substitute a newly generated split for saved historical membership without verification. Do not overwrite the author's final results with an older notebook output simply because that output is available here.

## Easiest asset handoff

Run `tools/collect_existing_run_assets.ipynb` in the Kaggle/Colab session or directory containing the saved outputs. It collects existing experiment metadata and records checkpoint locations without training. Include the checkpoints separately, or enable the explicit weight-copy option. If the original session is gone, attach the previously downloaded result ZIPs as inputs and point the collector at those specific folders. It can inventory ZIP contents without extracting or executing them.

The package can be uploaded as a transparent code snapshot. A claim of complete reproduction or publicly available trained weights must wait until the missing assets have been attached.
