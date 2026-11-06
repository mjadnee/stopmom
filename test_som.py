import contextlib
from dataclasses import replace
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigh
import torch

from SOM import SOM, build_som_grid
from SOM_config import CONFIG
import SOM_train as training


class PCAInitializationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.data = rng.normal(size=(80, 6)).astype(np.float32)
        self.tensor = torch.from_numpy(self.data)

    def test_initialization_matches_formula_for_each_geometry(self):
        for geometry in ("rectangular", "cylindrical", "toroidal"):
            with self.subTest(geometry=geometry):
                som = SOM(4, 6, 6, geometry=geometry)
                torch.manual_seed(17)
                _, singular, vectors = torch.pca_lowrank(
                    self.tensor, q=4 if geometry == "toroidal" else 3, center=False
                )
                lam = singular**2 / (len(self.tensor) - 1)
                x, y = som.xy.T
                u = 2 * math.pi * x / som.map_w
                b = 2 * y / y.max() - 1
                if geometry == "toroidal":
                    v = 2 * math.pi * y / (som.map_h * math.sqrt(3) / 2)
                    coef = torch.stack([
                        (2 * lam[0]).sqrt() * u.cos(),
                        (2 * lam[1]).sqrt() * u.sin(),
                        (2 * lam[2]).sqrt() * v.cos(),
                        (2 * lam[3]).sqrt() * v.sin(),
                    ], 1)
                elif geometry == "cylindrical":
                    coef = torch.stack([
                        (2 * lam[0]).sqrt() * u.cos(),
                        (2 * lam[1]).sqrt() * u.sin(),
                        (3 * lam[2]).sqrt() * b,
                    ], 1)
                else:
                    a = 2 * (x - x.min()) / (x.max() - x.min()) - 1
                    coef = torch.stack([
                        (3 * lam[0]).sqrt() * a,
                        (3 * lam[1]).sqrt() * b, torch.zeros_like(a),
                    ], 1)
                torch.manual_seed(17)
                som.init_pca(self.tensor)
                torch.testing.assert_close(
                    som.weights, (coef @ vectors.T).reshape(4, 6, 6), rtol=0, atol=0
                )
                if geometry == "toroidal":
                    weights = som.weights
                    steps = (
                        (weights.roll(-1, dims=1) - weights).square().sum(dim=2).mean(dim=1).sqrt()
                    )
                    torch.testing.assert_close(steps, steps[0].expand_as(steps))

    def test_grid_sizing_for_each_geometry(self):
        l1, l2, l3, l4 = eigh(self.data.T @ self.data, eigvals_only=True)[::-1][:4]
        ratios = {
            "rectangular": np.sqrt(l1 / l2),
            "cylindrical": min(2 * np.pi * np.sqrt((l1 + l2) / (12 * l3)), 4),
            "toroidal": np.sqrt((l1 + l2) / (l3 + l4)),
        }
        for geometry, ratio in ratios.items():
            with self.subTest(geometry=geometry):
                n = int(3 * np.sqrt(len(self.data)))
                height = round(np.sqrt(n / ratio))
                if geometry == "toroidal" and height % 2:
                    height += 1
                self.assertEqual(
                    build_som_grid(self.data, geometry), (height, round(n / height))
                )

    def test_toroidal_requires_four_features_and_even_height(self):
        with self.assertRaisesRegex(ValueError, "Ne >= 4"):
            SOM(4, 6, 3, geometry="toroidal").init_pca(self.tensor[:, :3])
        with self.assertRaisesRegex(ValueError, "Ne >= 4"):
            build_som_grid(self.data[:, :3], "toroidal")
        with self.assertRaisesRegex(ValueError, "even map_h"):
            SOM(3, 6, 6, geometry="toroidal")

    def test_toroidal_checkpoint_uses_geometry_name_and_reloads(self):
        with tempfile.TemporaryDirectory() as output:
            models = Path(output) / "fixture" / "models"
            models.mkdir(parents=True)
            config = replace(
                CONFIG, geometry="toroidal", map_shape=(4, 6),
                ordering_epochs=1, tuning_epochs=1, weighted=False,
            )
            args = (
                np.zeros((8, 10)), self.data.reshape(8, 10, 6), np.arange(6),
                np.arange(10), np.arange(8),
            )
            with (
                patch.object(training, "CONFIG", config),
                patch.object(training, "SOM_OUTPUT_DIR", output),
                patch.object(training, "load_som", wraps=training.load_som) as load,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                trained = training.train_som(*args, data_name="fixture", show_plots=False)
                load.assert_not_called()
                path = models / "fixture_4x6toroidal-SOM_.pt"
                self.assertTrue(path.exists())
                with patch.object(SOM, "fit", side_effect=AssertionError("Must reuse checkpoint")):
                    reloaded = training.train_som(*args, data_name="fixture", show_plots=False)
                load.assert_called_once()
                self.assertEqual(reloaded.geometry, "toroidal")
                torch.testing.assert_close(reloaded.weights, trained.weights)


if __name__ == "__main__":
    unittest.main()
