# Provenance

Mamadou implemented the original models and MLP training loop after studying
the theory through PyTorch documentation, Medium articles and Lilian Weng's
writing. These were implementations built from that understanding. Later Codex
engineering changes are recorded separately below.

The two Python files were copied from Mamadou's preexisting PyTorch learning
folder on 2 November 2026.

Codex prepared this copy and authored the README, dependency list, ignore rules
and this provenance note. In `cifar10_cnn.py`, Codex added a status docstring,
removed an unused import, adjusted spacing and redundant parentheses, and moved
the smoke-check input to the model's selected device. The device fix addresses
a runtime mismatch reproduced on CUDA in the original file.

Following Mamadou's explicit request to reuse the MLP training loop, Codex
adapted `train_loop` from his existing `kminst_mlp.py` into the shared
`train.py`. Codex added example-weighted metrics, deterministic
seeding, validation-based checkpoint selection, final test evaluation, plots,
CLI options and tests. Gradient clearing is performed before each forward pass;
the original MLP clears after each optimiser step. Both implement the same
gradient-reset cycle when starting with no accumulated gradients.

The CIFAR loader now accepts a shuffle seed and download flag; the 45,000/5,000
split and split seed remain unchanged. The model architecture, preprocessing,
SGD learning rate, momentum and weight decay are unchanged. A five-epoch
check was selected to verify the workflow, followed by the MLP script's original
20-epoch schedule. The five-epoch record is retained; neither run is an
optimisation claim. Measured results are recorded separately after execution. Original source
files outside this repository were left unchanged.

The standalone release uses MIT; dependency and dataset terms are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Standalone extraction

Extracted on 2 November 2026 from portfolio commit `6ad431f`.
Model definitions, data splits, optimisation and metric calculations are preserved.
Codex changed file paths and the CLI to run this dataset directly with `train.py`,
removed the dependency on the other project and set the default budget to 20 epochs.
Published metrics and curves were copied byte-for-byte from the recorded run.
The MIT licence was added at Mamadou's explicit request. Dependency and dataset
rights remain with their respective providers.

Learning references and dependency/dataset notices: see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The MIT licence does not
relicense the dependencies or downloaded datasets.

## Engineering review, 3 November 2026

At Mamadou's request, Codex authored the standalone `evaluate.py`, dataset-free
GitHub Actions workflow, additional checkpoint/preprocessing regression tests,
direct dependency version pins and tested-environment record, and the associated
README updates. The evaluator loads one explicitly supplied checkpoint without
training or checkpoint selection. Formatting changes are limited to the new
engineering code and tests.

All 10 dataset-free tests passed locally. CPU and CUDA evaluation of the existing
epoch-11 checkpoint both reproduced 50.22% validation and 50.43% test accuracy;
cross-entropy agreed with the original record within 0.000001. No confirmed bug
requiring a training-workflow change was found. The original model, data loader,
trainer, 20-epoch metrics/curves and five-epoch record were left byte-for-byte
unchanged. Local verification used an existing environment; a fresh local
installation was not tested. CI verification is reported separately by Actions.
