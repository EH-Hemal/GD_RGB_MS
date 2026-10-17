# Trained weights

No trained checkpoint is included in this package. Add the authors' actual downloaded checkpoints as release assets and record their URLs, SHA256, architecture, dataset, split, seed, preprocessing statistics and validation-selected threshold. Do not label ImageNet initialization weights as trained GD checkpoints.

Prefer best-model checkpoints; resume checkpoints also contain optimizer/RNG state and are larger. Load only trusted PyTorch checkpoint files. Strict state-dict loading is required; do not silently ignore mismatched keys or use a placeholder/random model for reproduction.
