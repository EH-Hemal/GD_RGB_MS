# Environment provenance

Original notebooks do not provide a complete immutable environment export. The explicit segmentation-models-pytorch pin is0.5.0; other original installer constraints are retained in requirements.txt. No package versions have been invented.

Run the asset collector in the original environment to record currently installed packages, Python, platform, GPU details and CMX git commit. A later export describes that later environment and must not be relabelled as the historical training environment. CUDA training, checkpoint loading and full-resolution inference were not tested during repository packaging because the packaging environment has no PyTorch/CUDA or dataset.
