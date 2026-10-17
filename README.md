# GD-RGBMS-Fusion

Research code for pixel-level rice grain discoloration segmentation using co-registered UAV RGB and multispectral images. The study compares modalities and fusion approaches, evaluates spatial transfer, and estimates image-level affected area.

**Release status:** curated source-code package. Saved trained weights, exact historical split files and several final-run exports have not yet been supplied. See [release status](docs/RELEASE_STATUS.md). This package does not claim that every manuscript result can already be reproduced from the attached assets.

## Start here

1. Obtain GD-RGBMS-DB from its original dataset release, or WeedyRice-RGBMS-DB for the separate weed task. See [data instructions](data/README.md).
2. Use a Kaggle GPU notebook, attach the dataset and upload ONE experiment notebook from `notebooks/`. The existing Kaggle paths are retained. Set the dataset root near the top if the mounted path differs.
3. For the proposed GD model, open `notebooks/01_proposed_standard_seed42.ipynb`. For the three pretrained baselines, use notebooks `09` and `10`. For buffered evaluation, use `13`.
4. Run the selected notebook from a fresh session. Preserve its output ZIP before ending the session. Some scripts install dependencies/download pretrained weights and require Internet access.
5. Read the explicitly labelled **no-TTA** test result for the main GD comparison. The proposed seed-42 source also saves a separate TTA experiment; its selected-TTA result is not the no-TTA result.

The training notebooks contain their full code; no hidden local training modules are required. Post-processing notebooks additionally require the documented checkpoints/CSV files. Start with the notebook descriptions in [experiment index](docs/EXPERIMENTS.md).

## Contents

| Folder | Contents |
|---|---|
| `notebooks/` | Separate training and plotting entrypoints |
| `models/` | Exact extracted proposed architecture, importable without training |
| `configs/` | Audited configuration expressions extracted from each source |
| `splits/` | Instructions for depositing exact historical memberships |
| `results/` | Reported summaries and selected original text outputs, with provenance |
| `figures/` | Supplied split figures; validation-membership caveat documented |
| `tools/` | Collect run assets, validate the package, and summarize measured per-seed CSVs |
| `docs/` | Experiment map, provenance, protocol details and unresolved assets |

## Environment

Use the original Kaggle/Colab training environment when available. `requirements.txt` lists the dependencies found in code; only versions explicitly specified in the sources are constrained. It is **not** an original environment lockfile. PyTorch/CUDA compatibility and CMX's original repository commit must be recorded from the training environment. See `environment/README.md`.

## Evaluation scope

The standard split, 24 m buffered split and bidirectional cross-acquisition experiments are separate protocols. The standard split is not a spatially independent field holdout. A 24 m camera-centre separation does not by itself establish non-overlapping image footprints. WeedyRice experiments retrain the architecture for weed segmentation and are not independent GD validation.

The proposed model was trained from scratch. SegFormer and CMX have slightly higher reported mean Dice. Parameter counts do not establish an inference-speed, memory or FLOP advantage. Training epoch caps differ for the pretrained seed-42 versus seed-123/2026 runs.

## Availability and attribution

Dataset authors retain their dataset rights; raw images are not bundled. Trained-weight download links are not yet available. Third-party model implementations and weights retain their original licenses. See [third-party attribution](THIRD_PARTY.md). The manuscript title, complete author list, software license and final repository/archival DOI should be supplied before a formal software release; no fictitious citation metadata or blanket third-party license is included.
