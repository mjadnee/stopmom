"""Generate a compact synthetic STM dataset for the SOM training example."""

import argparse
from pathlib import Path

import numpy as np


def generate_demo_stm(output_path):
    """Write a deterministic 256x256 STM dataset with 64 energy points."""
    rng = np.random.default_rng(7)
    coordinates = np.linspace(-1, 1, 256, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx)

    domain_ids = np.floor((angle + np.pi) * (6 / (2 * np.pi))).astype(int)
    domain_ids = np.clip(domain_ids, 0, 5)
    gaps_mv = np.array([14, 21, 28, 35, 42, 49], dtype=np.float32)
    gap = gaps_mv[domain_ids][..., None]
    bias = np.linspace(-0.1, 0.1, 64, dtype=np.float32)
    energy_mv = bias[None, None, :] * 1e3
    absolute_energy = np.abs(energy_mv)

    data = 0.18 + 0.72 / (1 + np.exp(-(absolute_energy - gap) / 3.5))
    data += 0.42 * np.exp(-0.5 * ((energy_mv - gap) / 5) ** 2)
    data += 0.36 * np.exp(-0.5 * ((energy_mv + gap) / 5) ** 2)
    data += (0.06 * np.sin(3 * angle) + 0.04 * radius)[..., None]
    data += rng.normal(0, 0.008, data.shape).astype(np.float32)
    data = data.astype(np.float32)

    lattice = np.cos(2 * np.pi * 12 * xx) + np.cos(2 * np.pi * 12 * yy)
    topography = 0.08 * lattice + 0.18 * np.sin(3 * angle) * np.exp(-radius**2)
    topography += rng.normal(0, 0.015, topography.shape)
    topography = topography.astype(np.float32)

    extent_m = 10e-9
    x = np.linspace(-extent_m / 2, extent_m / 2, coordinates.size, dtype=np.float32)
    y = np.linspace(-extent_m / 2, extent_m / 2, coordinates.size, dtype=np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        topography=topography,
        data=data,
        bias=bias,
        x=x,
        y=y,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("demo_stm.npz"),
        help="path for the generated NPZ archive",
    )
    args = parser.parse_args()
    print(f"Generated demo dataset: {generate_demo_stm(args.output)}")