# Data

Download the original datasets from their authors:

- GD-RGBMS-DB: https://doi.org/10.5281/zenodo.17008818
- WeedyRice-RGBMS-DB: https://doi.org/10.17632/vt4s83pxx6.1

GD code expects a root containing `RGB/`, `MS/`, `Mark/`, `train_list.txt`, `val_list.txt`, `test_list.txt`. The original Kaggle root is `/kaggle/input/datasets/hemal7/gd-rgbms-db`.

WeedyRice code searches beneath `/kaggle/input/datasets/hemal7/weedy-rgb-ms` for the directory containing split lists and `RGB/`, `Multispectral/`, `Masks/`.

Keep original image/sample names. Coordinates and altitude appear in filenames. Remove only recognised image extensions when matching names: using Path.stem on extension-free names ending in a decimal altitude can truncate the identifier. Do not redistribute raw images under a new license.
