# Preparation checks — 2 November 2026

Environment: Python 3.12; torch 2.14.0+cu126; torchvision 0.29.0+cu126;
NVIDIA GeForce RTX 4080 SUPER. These are observed installed versions, not an
independently verified package-installation recipe.

| Check | Observed result |
|---|---|
| Parse both Python files | Passed |
| Compare original/prepared model parameters with the same initialisation seed | Exact match |
| Compare original/prepared forward outputs on identical synthetic inputs | Exact match on CPU and CUDA |
| Synthetic output shape and finite-value checks | Passed: `(2, 10)` |
| Validation helper on synthetic inputs labelled by its predictions | Passed on CPU and CUDA; checks helper mechanics only |
| Compare original/prepared data-loader source | Byte-for-byte match |
| Load cached CIFAR-10 through `get_data_loaders` | Passed: 45,000 / 5,000 / 10,000 examples |
| Forward one actual training batch on CUDA | Passed: images `(64, 3, 32, 32)`, labels `(64,)`, finite logits `(64, 10)` |

The actual-data check called the existing loader with the cached dataset root,
took one batch, then ran `model(images.to(device))` under `torch.no_grad()`.
The public entry point uses the same operations with its default `./data` root.
Its network download path and a fresh environment installation were not tested.

The original smoke check failed on CUDA because images remained on CPU; the
prepared copy fixes that device mismatch. Data-folder access initially failed
inside the restricted environment; the later authorised check could read it.

These were the initial smoke checks before the training workflow was added.
Subsequently, the trainer was adapted from Mamadou's MLP loop. Four
dataset-free tests passed, covering parameter updates/checkpoint restoration,
example-weighted evaluation with no gradient/weight changes, empty evaluation
and non-finite training loss rejection.

The first complete CIFAR run used five epochs, seed 42 and the original
optimiser settings. Epoch 2 had the lowest validation loss; its held-out test
accuracy was 43.45% over 10,000 images. Its full record is retained in
`results/five_epoch_check.json`. A subsequent 20-epoch run follows the MLP
script's original training budget; its record is stored separately.

The 20-epoch run completed all 45,000 training and 5,000 validation examples
each epoch. Lowest validation cross-entropy selected epoch 11 (50.22%
validation accuracy), whose held-out accuracy was 50.43% on all 10,000 test
images, with test loss 1.4373996124. Both actual saved best checkpoints (CIFAR
and KMNIST) loaded on CPU and produced finite logits of the expected shape.
The first five CIFAR epoch records exactly match the earlier five-epoch run.
Both plots were visually inspected against their recorded curves.

The longer run used the original SGD settings and unchanged architecture,
seed 42, batch size 64 and deterministic algorithms. Python 3.12, torch
2.14.0+cu126 and torchvision 0.29.0+cu126 were installed. The shared dependency
list has not been tested in an isolated fresh installation; dataset downloads
were not needed because cached data were used.
