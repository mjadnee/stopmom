"""
Train a SOM from STM data arrays.
Edit "SOM_config.py" to adjust the map and training settings.
"""

import torch
import os
from pathlib import Path
import numpy as np

from SOM import SOM, load_som, save_som, build_som_grid
from SOM_config import CONFIG
from dataIO import load_data
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


def train_som(
    topography,
    data,
    bias,
    x,
    y,
    data_name="stm_dataset",
    save_images=False,
    interactive_plots=False,
    n_superclusters=None,
    save_clusters=False,
):

    topography, data, bias, x, y = _validate_stm_data(topography, data, bias, x, y)
    if save_clusters and n_superclusters is None:
        raise ValueError("n_superclusters must be set when save_clusters is enabled")

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

    if save_clusters:
        normalized_data = flat_data.reshape(data.shape)
        _, cluster_ids = plots.get_supercluster_labels(normalized_data, som, n_superclusters)
        cluster_dir = os.path.join(SOM_OUTPUT_DIR, data_name, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_path = os.path.join(
            cluster_dir,
            f"{save_name}_{n_superclusters}_superclusters.npz",
        )
        np.savez_compressed(
            cluster_path,
            topography=topography,
            data=data,
            bias=bias,
            x=x,
            y=y,
            supercluster_ids=cluster_ids,
        )
        print(f"Saved supercluster-labeled data to {cluster_path}")

    if save_images or interactive_plots:
        os.makedirs(figure_dir, exist_ok=True)
        plot_context = {
            "figure_dir": figure_dir,
            "save_name": save_name,
            "save_images": save_images,
            "data_name": data_name,
        }
        normalized_data = flat_data.reshape(data.shape)
        plots.plot_som_results(topography, normalized_data, bias, x, y, som, plot_context)
        plots.plot_som_error_maps(topography, normalized_data, bias, x, y, som, plot_context)
        if interactive_plots:
            plots.plot_interactive_som_results(topography, normalized_data, bias, x, y, som, plot_context=plot_context)
        if n_superclusters is not None:
            plots.plot_som_superclusters(
                topography, normalized_data, bias, x, y, som, n_superclusters, plot_context
            )
            if interactive_plots:
                plots.plot_interactive_som_superclusters(
                    topography, normalized_data, bias, x, y, som, n_superclusters, plot_context=plot_context
                )
    return som


def main():
    file_ext = CONFIG.file_type if CONFIG.file_type.startswith(".") else f".{CONFIG.file_type}"
    file_name = f"{CONFIG.data_name}{file_ext}"

    if os.path.isabs(CONFIG.data_path):
        full_file_path = os.path.join(CONFIG.data_path, file_name)
    else:
        full_file_path = str(Path(__file__).resolve().parent / CONFIG.data_path / file_name)

    if file_ext == ".npz":
        with np.load(full_file_path) as datacube:
            topography, data, bias, x, y = (
                datacube[name] for name in ("topography", "data", "bias", "x", "y")
            )
    elif file_ext == ".3ds":
        topography, data, bias, x, y = load_data(full_file_path)
    else:
        raise ValueError(f"Unsupported file type '{CONFIG.file_type}'. Must be '.npz' or '.3ds'.")

    train_som(
        topography,
        data,
        bias,
        x,
        y,
        data_name=CONFIG.data_name,
        save_images=CONFIG.save_imgs,
        interactive_plots=CONFIG.interactive_plots,
        n_superclusters=CONFIG.n_superclusters,
        save_clusters=CONFIG.save_clusters,
    )


if __name__ == "__main__":
    main()
