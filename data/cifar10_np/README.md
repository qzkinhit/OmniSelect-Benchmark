# CIFAR-10 arrays for the native protocol

`cifar10.npz` holds `Xtr` (50,000 x 32 x 32 x 3 uint8), `ytr`, `Xte` (10,000), `yte`, decoded from
Hugging Face `uoft-cs/cifar10` at revision `0b2714987fa4...`. `python data/fetch_data.py --only hf`
writes it and checks the content hash of every array. The native track repeats the check.
The file is 185 MB and ignored by Git.
