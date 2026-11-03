"""Dataset-free evaluation, checkpoint and preprocessing regression checks."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import cifar10_data
from cifar10_cnn import CIFAR10CNN
import evaluate


class EvaluationTests(unittest.TestCase):
    def test_short_batch_is_weighted_and_model_is_not_updated(self):
        torch.manual_seed(7)
        model = nn.Linear(2, 2)
        images = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        labels = torch.tensor([0, 1, 0])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        expected_loss = nn.CrossEntropyLoss()(model(images), labels).item()
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        result = evaluate.evaluate_split(loader, model, "cpu")
        self.assertEqual(result["examples"], 3)
        self.assertAlmostEqual(result["loss"], expected_loss, places=6)
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_cli_loads_one_checkpoint_and_never_trains(self):
        torch.manual_seed(7)
        model = CIFAR10CNN().eval()
        images = torch.rand(3, 3, 32, 32) * 2 - 1
        with torch.no_grad():
            labels = model(images).argmax(1)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            output = Path(directory) / "evaluation.json"
            torch.save({"dataset": "cifar10", "epoch": 11, "seed": 42,
                        "model_state_dict": model.state_dict()}, checkpoint)
            original = checkpoint.read_bytes()
            argv = ["evaluate.py", "--checkpoint", str(checkpoint), "--device", "cpu",
                    "--output", str(output)]
            # The first returned object cannot be iterated as a training loader.
            with patch("sys.argv", argv), \
                 patch.object(evaluate, "get_data_loaders", return_value=(object(), loader, loader)) as get_loaders, \
                 patch("torch.optim.SGD", side_effect=AssertionError("Evaluation must not train")), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            get_loaders.assert_called_once_with(root="data", batch_size=64, seed=42, download=False)
            result = json.loads(output.read_text())
            self.assertEqual(result["checkpoint_epoch"], 11)
            self.assertEqual(result["test"]["accuracy_percent"], 100.0)
            self.assertEqual(result["validation"]["examples"], 3)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_incompatible_model_state_fails_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wrong-shape.pt"
            torch.save({"dataset": "cifar10", "epoch": 11, "seed": 42,
                        "model_state_dict": {}}, checkpoint)
            with patch("sys.argv", ["evaluate.py", "--checkpoint", str(checkpoint), "--device", "cpu"]), \
                 patch.object(evaluate, "get_data_loaders", side_effect=AssertionError("No data needed")), \
                 self.assertRaises(RuntimeError):
                evaluate.main()

    def test_existing_preprocessing_and_labels_are_shared_across_splits(self):
        class FakeCIFAR:
            def __init__(self, root, train, download, transform):
                self.train = train
                self.transform = transform
                self.download = download

            def __len__(self):
                return 50000 if self.train else 10000

            def __getitem__(self, index):
                image = Image.fromarray(np.full((32, 32, 3), (index % 2) * 255, dtype=np.uint8))
                return self.transform(image), index % 10

        with patch.object(cifar10_data.datasets, "CIFAR10", FakeCIFAR):
            train, validation, test = cifar10_data.get_data_loaders(download=False)
        self.assertIs(train.dataset.dataset, validation.dataset.dataset)
        self.assertIs(train.dataset.dataset.transform, test.dataset.transform)
        self.assertFalse(test.dataset.train)
        self.assertFalse(test.dataset.download)
        for dataset in [train.dataset.dataset, test.dataset]:
            black, label0 = dataset[0]
            white, label1 = dataset[1]
            self.assertEqual(black.shape, (3, 32, 32))
            self.assertTrue(torch.all(black == -1))
            self.assertTrue(torch.all(white == 1))
            self.assertEqual((label0, label1), (0, 1))


if __name__ == "__main__":
    unittest.main()
