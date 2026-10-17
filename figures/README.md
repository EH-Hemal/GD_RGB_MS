# Supplied spatial figures

These PNGs were supplied by the author after running plotting code. Their visual appearance alone cannot verify the historical training memberships.

- buffered24m_single_panel.png:207/40/41/112 camera centres. Match to saved training protocol.
- cross_acquisition_membership.png: final all-source refit versus unseen target. This does not show the earlier source validation partition.
- cross_acquisition_train_val_test_seed42.png: development partition illustration. Its original plotting default uses NumPy for B-to-A, while the later B-to-A training runner uses sklearn. Regenerate from actual saved run lists before using this version as the training-membership figure.
