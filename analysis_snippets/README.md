# Coverage analysis

`coverage_seed42.py` is the original cell 53. It depends on an already loaded final model, test_loader, device, MEAN and STD in the SAME notebook session. It uses the saved validation-selected threshold 0.60. It is not a standalone training/evaluation entrypoint. Run after restoring the appropriate seed-42 model, never after an unrelated experiment. The original seed-42 analysis code for Wilcoxon/Holm/bootstrap was not found; no replacement is presented as the historical analysis.
