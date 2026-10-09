import torch
import os
import numpy as np

from SOM import SOM, load_som, save_som, build_som_grid
from SOM_config import CONFIG
import SOM_plots as plots

SOM_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SOM_outputs")

def zero_bias_feature_weights(bias, width):
    """Return feature weights that emphasize spectra closest to zero bias."""
    if width <= 0:
        raise ValueError("width must be positive")
    return 1 + 4 * np.exp(-0.5 * (bias / width) ** 2)


def _validate_stm_data(topography, data, bias, x, y):
    topography = np.asarray(topography, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("data must have shape (y_pixels, x_pixels, energy_points)")
    if topography.shape != data.shape[:2]:
        raise ValueError("topography must have shape data.shape[:2]")
    if bias.shape != (data.shape[2],):
        raise ValueError("bias must have one value per energy point")
    if x.shape != (data.shape[1],) or y.shape != (data.shape[0],):
        raise ValueError("x and y must match the data x and y dimensions")
    if not all(np.isfinite(array).all() for array in (topography, data, bias, x, y)):
        raise ValueError("STM input arrays must contain only finite values")
    return topography, data, bias, x, y


def train_som(topography, data, bias, x, y, data_name="stm_dataset", save_images=False,
              interactive_plots=False, n_superclusters=None):
    """Train a SOM from an STM data cube and optional coordinate arrays.

    ``data`` has shape ``(y_pixels, x_pixels, energy_points)``. ``topography``
    has shape ``(y_pixels, x_pixels)``, while ``bias``, ``x``, and ``y`` are
    one-dimensional arrays for the energy, x, and y axes respectively.
    """
    topography, data, bias, x, y = _validate_stm_data(topography, data, bias, x, y)
    flat_unnorm_data = data.reshape(-1, data.shape[-1])
    standard_deviation = flat_unnorm_data.std(axis=1, keepdims=True)
    if np.any(standard_deviation == 0):
        raise ValueError("each STM spectrum must have non-zero variance")
    flat_data = (flat_unnorm_data - flat_unnorm_data.mean(axis=1, keepdims=True)) / standard_deviation
    flat_data_tensor = torch.from_numpy(flat_data)
    feature_weights = zero_bias_feature_weights(bias, CONFIG.zero_bias_width) if CONFIG.weighted else None
    map_h, map_w = CONFIG.map_shape or build_som_grid(flat_data, geometry=CONFIG.geometry)
    print(f"Using {CONFIG.geometry} SOM geometry {map_h}x{map_w}")

    weighting_suffix = "_weighted" if CONFIG.weighted else ""
    save_name = f"{data_name}_{map_h}x{map_w}{CONFIG.geometry}-SOM_{weighting_suffix}"
    model_dir = os.path.join(SOM_OUTPUT_DIR, data_name, "models")
    figure_dir = os.path.join(SOM_OUTPUT_DIR, data_name, "figures")
    som_path = os.path.join(model_dir, save_name + ".pt")

    if os.path.exists(som_path):
        print(f"Loading existing SOM: {save_name}")
        som = load_som(som_path, device="cpu")
    else:
        print("Training new SOM...")
        som = SOM(
            map_h,
            map_w,
            data.shape[-1],
            geometry=CONFIG.geometry,
            seed=CONFIG.seed,
            feature_weights=feature_weights,
        )
        som.fit(
            flat_data_tensor,
            ordering_epochs=CONFIG.ordering_epochs,
            tuning_epochs=CONFIG.tuning_epochs,
            sigma_end=CONFIG.sigma_end,
            subset_fraction=CONFIG.subset_fraction,
        )
        os.makedirs(model_dir, exist_ok=True)
        save_som(som, som_path, hyperparameters=CONFIG.checkpoint_metadata())
        print(f"Model saved: {save_name}")

    if save_images or interactive_plots:
        os.makedirs(figure_dir, exist_ok=True)
        plot_context = {"figure_dir": figure_dir, "save_name": save_name, "save_images": save_images}
        normalized_data = flat_data.reshape(data.shape)
        plots.plot_som_results(topography, normalized_data, bias, x, y, som, plot_context)
        plots.plot_som_error_maps(topography, normalized_data, bias, x, y, som, plot_context)
        if interactive_plots:
            plots.plot_interactive_som_results(topography, normalized_data, bias, x, y, som)
        if n_superclusters is not None:
            plots.plot_som_superclusters(
                topography, normalized_data, bias, x, y, som, n_superclusters, plot_context
            )
            if interactive_plots:
                plots.plot_interactive_som_superclusters(
                    topography, normalized_data, bias, x, y, som, n_superclusters
                )
    return som


if __name__ == "__main__":
    raise SystemExit("Import train_som and pass your STM data arrays. See README.md for the required array shapes.")
