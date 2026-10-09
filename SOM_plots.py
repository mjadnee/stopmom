import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import Slider


from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix

ERROR_CMAP = "magma"
ERROR_LIMITS = (0, 100)


def plot_som_hexagons(axis, som, values=None, cmap="viridis", norm=None,
                     vmin=None, vmax=None, edgecolors="none", linewidths=0, facecolors=None):
    """Render one value per SOM node as a hexagon centered at som.xy."""
    centers = som.xy.detach().cpu().numpy()
    radius = 1 / np.sqrt(3)
    angles = np.arange(6) * np.pi / 3 + np.pi / 6
    vertices = centers[:, None, :] + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    collection = PolyCollection(vertices, edgecolors=edgecolors, linewidths=linewidths, facecolors=facecolors)
    if facecolors is None:
        collection.set_array(np.ma.asarray(values).reshape(-1))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_clim(vmin, vmax)
    axis.add_collection(collection)
    axis.set_xlim(centers[:, 0].min() - radius, centers[:, 0].max() + radius)
    axis.set_ylim(centers[:, 1].min() - radius, centers[:, 1].max() + radius)
    axis.set_aspect("equal")
    return collection

def som_position_colors(som):
    """Return node colors with hue set by column and brightness set by row."""
    rows = np.repeat(np.arange(som.map_h), som.map_w)
    columns = np.tile(np.arange(som.map_w), som.map_h)
    hue = columns / som.map_w
    brightness = 0.35 + 0.65 * rows / max(som.map_h - 1, 1)
    return mcolors.hsv_to_rgb(np.column_stack((hue, np.full_like(hue, 0.8), brightness)))

def cluster_som_nodes(som, n_superclusters):
    """Cluster prototypes while permitting merges only between hex neighbors."""
    n_nodes = som.map_h * som.map_w
    if not 2 <= n_superclusters <= n_nodes:
        raise ValueError(f"n_superclusters must be between 2 and {n_nodes}")

    prototypes = som.weights.detach().cpu().numpy().reshape(n_nodes, som.Ne)
    feature_scale = som.feature_weights.detach().cpu().numpy() ** 0.5
    neighbors = (som.D2.detach().cpu().numpy() > 0) & (som.D2.detach().cpu().numpy() < 1.5)
    return AgglomerativeClustering(
        n_clusters=n_superclusters,
        linkage="ward",
        connectivity=csr_matrix(neighbors),
    ).fit_predict(prototypes * feature_scale)

def plot_bmu_samples(axis, bias, spectra, rng, color="tab:blue"):
    """Plot up to 100 transparent spectra assigned to a selected BMU."""
    if len(spectra) == 0:
        return []
    samples = spectra[rng.choice(len(spectra), size=min(100, len(spectra)), replace=False)]
    return axis.plot(bias, samples.T, color=color, alpha=0.5, linewidth=0.75)






def plot_som_results(topo, data, bias, x, y, som, plot_context):
    """Plot topography, zero-bias dI/dV, SOM positions, U-matrix, and node QE."""
    save_path = os.path.join(plot_context["figure_dir"], f"{plot_context['save_name']}_results.png") if plot_context["save_images"] else None
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    bias_mv = bias * 1e3
    x_nm = x * 1e9
    y_nm = y * 1e9
    samples = torch.from_numpy(data.reshape(-1, data.shape[-1])).to(som.device, torch.float32)
    n_nodes = som.map_h * som.map_w
    weights = som.weights.reshape(n_nodes, som.Ne)
    feature_weights = som.feature_weights
    weight_norms = (weights * weights * feature_weights).sum(1)
    hits = torch.zeros(n_nodes, device=som.device)
    qe_sums = torch.zeros(n_nodes, device=som.device)
    bmu_chunks = []
    for sample_chunk in samples.split(16384):
        weighted_samples = sample_chunk * feature_weights
        distances, bmu_ids = torch.addmm(weight_norms, weighted_samples, weights.T, alpha=-2).min(1)
        signal_norm_squared = (sample_chunk * sample_chunk * feature_weights).sum(1)
        qe = 100 * (distances + signal_norm_squared).clamp_min(0) / signal_norm_squared.clamp_min(
            torch.finfo(samples.dtype).eps
        )
        hits.index_add_(0, bmu_ids, torch.ones_like(qe))
        qe_sums.index_add_(0, bmu_ids, qe)
        bmu_chunks.append(bmu_ids)
    qe_map = torch.where(hits > 0, qe_sums / hits, torch.nan).view(som.map_h, som.map_w).cpu().numpy()
    bmu_ids = torch.cat(bmu_chunks).view(data.shape[:2]).cpu().numpy()
    node_colors = som_position_colors(som)
    color_key = mcolors.ListedColormap(node_colors)
    color_norm = mcolors.BoundaryNorm(np.arange(n_nodes + 1) - 0.5, n_nodes)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    topo_min, topo_max = np.percentile(topo, (1, 99))
    image = axes[0, 0].pcolormesh(x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max, shading="auto")
    axes[0, 0].set_title("Topography")
    axes[0, 0].set_xlabel("X (nm)")
    axes[0, 0].set_ylabel("Y (nm)")
    axes[0, 0].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 0], label="Topography")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    data_min, data_max = np.percentile(zero_bias_slice, (1, 99))
    image = axes[0, 1].pcolormesh(x_nm, y_nm, zero_bias_slice, cmap="gray", vmin=data_min, vmax=data_max, shading="auto")
    axes[0, 1].set_title(f"Normalized dI/dV at {bias_mv[zero_bias_index]:.1f} mV")
    axes[0, 1].set_xlabel("X (nm)")
    axes[0, 1].set_ylabel("Y (nm)")
    axes[0, 1].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 1], label="Normalized signal")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    axes[0, 2].pcolormesh(x_nm, y_nm, bmu_ids, cmap=color_key, norm=color_norm, shading="auto")
    axes[0, 2].set_title("SOM Position (Real Space)")
    axes[0, 2].set_xlabel("X (nm)")
    axes[0, 2].set_ylabel("Y (nm)")
    axes[0, 2].set_aspect("equal")

    image = plot_som_hexagons(
        axes[1, 0], som, np.ma.masked_invalid(qe_map), cmap=ERROR_CMAP,
        vmin=ERROR_LIMITS[0], vmax=ERROR_LIMITS[1]
    )
    axes[1, 0].set_title("Relative QE (SOM)")
    axes[1, 0].set_xlabel("SOM column")
    axes[1, 0].set_ylabel("SOM row")
    colorbar = fig.colorbar(image, ax=axes[1, 0], label="Relative QE (%)", shrink=0.65, aspect=18, pad=0.03)
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    image = plot_som_hexagons(axes[1, 1], som, som.get_u_matrix(), cmap="viridis")
    axes[1, 1].set_title("U-Matrix")
    axes[1, 1].set_xlabel("SOM column")
    axes[1, 1].set_ylabel("SOM row")
    colorbar = fig.colorbar(image, ax=axes[1, 1], label="Mean neighbor distance", shrink=0.65, aspect=18, pad=0.03)
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    plot_som_hexagons(axes[1, 2], som, facecolors=node_colors, edgecolors="white", linewidths=0.2)
    axes[1, 2].set_title("SOM Position Color Guide")
    axes[1, 2].set_xlabel("SOM column")
    axes[1, 2].set_ylabel("SOM row")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved diagnostics plot to {save_path}")
    else:
        plt.show()

def plot_som_superclusters(topo, data, bias, x, y, som, n_superclusters, plot_context):
    """Plot real-space and SOM superclusters with their weighted variance explained."""
    n_nodes = som.map_h * som.map_w
    supercluster_labels = cluster_som_nodes(som, n_superclusters)
    bmus = som.predict(torch.from_numpy(data.reshape(-1, data.shape[-1])))
    bmu_ids = bmus[:, 0].numpy() * som.map_w + bmus[:, 1].numpy()
    real_space_labels = supercluster_labels[bmu_ids].reshape(data.shape[:2])
    x_nm = x * 1e9
    y_nm = y * 1e9
    label_grid = supercluster_labels.reshape(som.map_h, som.map_w)
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    spectra = data.reshape(-1, data.shape[-1])
    labels = real_space_labels.ravel()
    counts = np.bincount(labels, minlength=n_superclusters)
    okabe_ito = np.array([
        mcolors.to_rgb(color) for color in (
            "#E69F00", "#56B4E9", "#009E73", "#F0E442",
            "#0072B2", "#D55E00", "#CC79A7", "#000000",
        )
    ])
    extra_colors = plt.get_cmap("tab20").colors
    palette = np.vstack((okabe_ito, np.resize(extra_colors, (max(n_superclusters - 8, 0), 3))))
    count_order = np.argsort(-counts, kind="stable")
    color_rank = np.empty(n_superclusters, dtype=int)
    color_rank[count_order] = np.arange(n_superclusters)
    cmap = mcolors.ListedColormap(palette)
    norm = mcolors.BoundaryNorm(np.arange(n_superclusters + 1) - 0.5, n_superclusters)
    real_space_ranks = color_rank[real_space_labels]
    node_ranks = color_rank[label_grid]
    cluster_sums = np.zeros((n_superclusters, data.shape[-1]), dtype=spectra.dtype)
    np.add.at(cluster_sums, labels, spectra)
    cluster_means = cluster_sums / counts[:, None]
    feature_weights = som.feature_weights.detach().cpu().numpy()
    total_variation = ((spectra - spectra.mean(axis=0)) ** 2 * feature_weights).sum()
    residual_variation = ((spectra - cluster_means[labels]) ** 2 * feature_weights).sum()
    variance_explained = 1 - residual_variation / total_variation if total_variation > 0 else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    topo_min, topo_max = np.percentile(topo, (1, 99))
    image = axes[0, 0].pcolormesh(x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max, shading="auto")
    axes[0, 0].set_title("Topography")
    axes[0, 0].set_xlabel("X (nm)")
    axes[0, 0].set_ylabel("Y (nm)")
    axes[0, 0].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 0], label="Topography")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    didv_min, didv_max = np.percentile(zero_bias_slice, (1, 99))
    image = axes[0, 1].pcolormesh(
        x_nm, y_nm, zero_bias_slice, cmap="gray", vmin=didv_min, vmax=didv_max, shading="auto"
    )
    axes[0, 1].set_title(f"Normalized dI/dV at {bias[zero_bias_index] * 1e3:.1f} mV")
    axes[0, 1].set_xlabel("X (nm)")
    axes[0, 1].set_ylabel("Y (nm)")
    axes[0, 1].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 1], label="Normalized signal")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    image = axes[0, 2].pcolormesh(x_nm, y_nm, real_space_ranks, cmap=cmap, norm=norm, shading="auto")
    axes[0, 2].set_title("Superclusters (Real Space, Largest to Smallest)")
    axes[0, 2].set_xlabel("X (nm)")
    axes[0, 2].set_ylabel("Y (nm)")
    axes[0, 2].set_aspect("equal")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.5, f"Variance Explained\n{variance_explained:.2%}",
        ha="center", va="center", fontsize=18, transform=axes[1, 0].transAxes,
    )

    image = plot_som_hexagons(axes[1, 1], som, som.get_u_matrix(), cmap="viridis")
    axes[1, 1].set_title("U-Matrix")
    axes[1, 1].set_xlabel("SOM column")
    axes[1, 1].set_ylabel("SOM row")
    colorbar = fig.colorbar(image, ax=axes[1, 1], label="Mean neighbor distance", shrink=0.65, aspect=18, pad=0.03)
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    plot_som_hexagons(axes[1, 2], som, node_ranks, cmap=cmap, norm=norm, edgecolors="white", linewidths=0.4)
    axes[1, 2].set_title("Superclusters (SOM, Largest to Smallest)")
    axes[1, 2].set_xlabel("SOM column")
    axes[1, 2].set_ylabel("SOM row")

    save_path = os.path.join(
        plot_context["figure_dir"],
        f"{plot_context['save_name']}_{n_superclusters}_superclusters.png",
    ) if plot_context["save_images"] else None
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved supercluster plot to {save_path}")
    else:
        plt.show()

    return supercluster_labels, counts

def plot_som_error_maps(topo, data, bias, x, y, som, plot_context):
    """Plot real-space and SOM-space relative QE with a map-wide QE/TE summary."""
    x_nm = x * 1e9
    y_nm = y * 1e9
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    samples = torch.from_numpy(data.reshape(-1, data.shape[-1])).to(som.device, torch.float32)
    n_nodes = som.map_h * som.map_w
    weights = som.weights.reshape(n_nodes, som.Ne)
    feature_weights = som.feature_weights
    weight_norms = (weights * weights * feature_weights).sum(1)
    hits = torch.zeros(n_nodes, device=som.device)
    qe_sums = torch.zeros(n_nodes, device=som.device)
    te_sums = torch.zeros(n_nodes, device=som.device)
    qe_chunks = []
    for sample_chunk in samples.split(16384):
        weighted_samples = sample_chunk * feature_weights
        distances, indices = torch.addmm(weight_norms, weighted_samples, weights.T, alpha=-2).topk(
            2, dim=1, largest=False
        )
        bmu_ids = indices[:, 0]
        signal_norm_squared = (sample_chunk * sample_chunk * feature_weights).sum(1)
        qe = 100 * (distances[:, 0] + signal_norm_squared).clamp_min(0) / signal_norm_squared.clamp_min(
            torch.finfo(samples.dtype).eps
        )
        te = 100 * (som.D2[indices[:, 0], indices[:, 1]] > 1.5).to(samples.dtype)
        hits.index_add_(0, bmu_ids, torch.ones_like(qe))
        qe_sums.index_add_(0, bmu_ids, qe)
        te_sums.index_add_(0, bmu_ids, te)
        qe_chunks.append(qe)
    qe_map = torch.where(hits > 0, qe_sums / hits, torch.nan).view(som.map_h, som.map_w).cpu().numpy()
    te_map = torch.where(hits > 0, te_sums / hits, torch.nan).view(som.map_h, som.map_w).cpu().numpy()
    qe_real_space = torch.cat(qe_chunks).view(data.shape[:2]).cpu().numpy()
    mean_qe = qe_sums.sum().item() / len(samples)
    mean_te = te_sums.sum().item() / len(samples)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    topo_min, topo_max = np.percentile(topo, (1, 99))
    image = axes[0, 0].pcolormesh(x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max, shading="auto")
    axes[0, 0].set_title("Topography")
    axes[0, 0].set_xlabel("X (nm)")
    axes[0, 0].set_ylabel("Y (nm)")
    axes[0, 0].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 0], label="Topography")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    didv_min, didv_max = np.percentile(zero_bias_slice, (1, 99))
    image = axes[0, 1].pcolormesh(
        x_nm, y_nm, zero_bias_slice, cmap="gray", vmin=didv_min, vmax=didv_max, shading="auto"
    )
    axes[0, 1].set_title(f"Normalized dI/dV at {bias[zero_bias_index] * 1e3:.1f} mV")
    axes[0, 1].set_xlabel("X (nm)")
    axes[0, 1].set_ylabel("Y (nm)")
    axes[0, 1].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 1], label="Normalized signal")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    image = axes[0, 2].pcolormesh(
        x_nm, y_nm, qe_real_space, cmap=ERROR_CMAP, vmin=ERROR_LIMITS[0], vmax=ERROR_LIMITS[1], shading="auto"
    )
    axes[0, 2].set_title("Relative QE (Real Space)")
    axes[0, 2].set_xlabel("X (nm)")
    axes[0, 2].set_ylabel("Y (nm)")
    axes[0, 2].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 2], label="Relative QE (%)")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.5, f"Map-Wide Relative QE\n{mean_qe:.2f}%\n\nMap-Wide TE\n{mean_te:.2f}%",
        ha="center", va="center", fontsize=16, transform=axes[1, 0].transAxes,
    )

    image = plot_som_hexagons(
        axes[1, 1], som, np.ma.masked_invalid(te_map), cmap=ERROR_CMAP,
        vmin=ERROR_LIMITS[0], vmax=ERROR_LIMITS[1]
    )
    axes[1, 1].set_title("TE (SOM)")
    axes[1, 1].set_xlabel("SOM column")
    axes[1, 1].set_ylabel("SOM row")
    colorbar = fig.colorbar(image, ax=axes[1, 1], label="TE (%)", shrink=0.65, aspect=18, pad=0.03)
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    image = plot_som_hexagons(
        axes[1, 2], som, np.ma.masked_invalid(qe_map), cmap=ERROR_CMAP,
        vmin=ERROR_LIMITS[0], vmax=ERROR_LIMITS[1]
    )
    axes[1, 2].set_title("Relative QE (SOM)")
    axes[1, 2].set_xlabel("SOM column")
    axes[1, 2].set_ylabel("SOM row")
    colorbar = fig.colorbar(image, ax=axes[1, 2], label="Relative QE (%)", shrink=0.65, aspect=18, pad=0.03)
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()
    save_path = os.path.join(plot_context["figure_dir"], f"{plot_context['save_name']}_node_error_maps.png") if plot_context["save_images"] else None
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()




def plot_interactive_som_superclusters(topo, data, bias, x, y, som, n_superclusters):
    """Explore real-space spectra and connected supercluster centroids by clicking maps."""
    n_nodes = som.map_h * som.map_w
    max_clusters = min(n_nodes, 15)
    n_superclusters = min(n_superclusters, max_clusters)
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    bias_mv = bias * 1e3
    supercluster_labels = cluster_som_nodes(som, n_superclusters)

    bmus = som.predict(torch.from_numpy(data.reshape(-1, data.shape[-1])))
    bmu_ids = bmus[:, 0].numpy() * som.map_w + bmus[:, 1].numpy()
    real_space_labels = supercluster_labels[bmu_ids].reshape(data.shape[:2])
    label_grid = supercluster_labels.reshape(som.map_h, som.map_w)
    som_coordinates = som.xy.detach().cpu().numpy()
    x_nm = x * 1e9
    y_nm = y * 1e9
    spectra = data.reshape(-1, data.shape[-1])
    feature_weights = som.feature_weights.detach().cpu().numpy()
    total_variation = ((spectra - spectra.mean(axis=0)) ** 2 * feature_weights).sum()

    def variance_explained(pixel_labels, n_clusters):
        counts = np.bincount(pixel_labels, minlength=n_clusters)
        sums = np.zeros((n_clusters, data.shape[-1]), dtype=spectra.dtype)
        np.add.at(sums, pixel_labels, spectra)
        means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
        residual_variation = ((spectra - means[pixel_labels]) ** 2 * feature_weights).sum()
        return 1 - residual_variation / total_variation if total_variation > 0 else np.nan

    cmap = plt.get_cmap("tab20", n_superclusters)

    norm = mcolors.BoundaryNorm(np.arange(n_superclusters + 1) - 0.5, n_superclusters)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.15, top=0.94, wspace=0.3, hspace=0.3)
    topo_axis, data_axis, variance_axis, som_axis, spectrum_axis, u_matrix_axis = axes.ravel()
    topo_min, topo_max = np.percentile(topo, (1, 99))
    topo_axis.pcolormesh(x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max, alpha=0.45, shading="auto")
    topo_axis.set_title("Topography")
    topo_axis.set_xlabel("X (nm)")
    topo_axis.set_ylabel("Y (nm)")
    topo_axis.set_aspect("equal")
    selected_topography_pixels = topo_axis.pcolormesh(
        x_nm,
        y_nm,
        np.ma.masked_all(real_space_labels.shape),
        cmap=mcolors.ListedColormap(["black"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
        shading="auto",
    )

    data_min, data_max = np.percentile(zero_bias_slice, (1, 99))
    data_axis.pcolormesh(
        x_nm,
        y_nm,
        zero_bias_slice,
        cmap="gray",
        vmin=data_min,
        vmax=data_max,
        alpha=0.45,
        shading="auto",
    )
    data_axis.set_title(f"Normalized dI/dV at {bias_mv[zero_bias_index]:.1f} mV")
    data_axis.set_xlabel("X (nm)")
    data_axis.set_ylabel("Y (nm)")
    data_axis.set_aspect("equal")
    selected_pixels = data_axis.pcolormesh(
        x_nm,
        y_nm,
        np.ma.masked_all(real_space_labels.shape),
        cmap=mcolors.ListedColormap(["black"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
        shading="auto",
    )

    variance_axis.axis("off")
    variance_text = variance_axis.text(
        0.5,
        0.5,
        f"Variance Explained\n{variance_explained(real_space_labels.ravel(), n_superclusters):.2%}",
        ha="center",
        va="center",
        fontsize=18,
        transform=variance_axis.transAxes,
    )

    som_collection = plot_som_hexagons(som_axis, som, label_grid, cmap=cmap, norm=norm,
                                       edgecolors="white", linewidths=0.4)
    som_axis.set_title("SOM Superclusters")
    som_axis.set_xlabel("SOM column")
    som_axis.set_ylabel("SOM row")

    image = plot_som_hexagons(u_matrix_axis, som, som.get_u_matrix(), cmap="viridis")
    u_matrix_axis.set_title("U-Matrix")
    u_matrix_axis.set_xlabel("SOM column")
    u_matrix_axis.set_ylabel("SOM row")
    colorbar = fig.colorbar(image, ax=u_matrix_axis, label="Mean neighbor distance", shrink=0.65, aspect=18, pad=0.03)
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    spectrum_axis.set_title("Selected Spectrum")
    spectrum_axis.set_xlabel("Bias (mV)")
    spectrum_axis.set_ylabel("Normalized dI/dV")
    spectrum_axis.set_xlim(bias_mv.min(), bias_mv.max())
    spectrum_axis.set_ylim(-5, 5)
    spectrum_axis.grid(alpha=0.3)
    spectrum_axis.axvline(0, color="0.25", linewidth=1.25, zorder=0)
    spectrum_line, = spectrum_axis.plot([], [], color="black", zorder=3)
    sample_lines = []
    rng = np.random.default_rng()
    selection_label = spectrum_axis.text(
        0.02, 0.98, "Click a real-space pixel or SOM node", transform=spectrum_axis.transAxes, va="top"
    )

    data_marker, = data_axis.plot([], [], "wo", markeredgecolor="black", markersize=7)
    som_marker, = som_axis.plot([], [], "wo", markeredgecolor="black", markersize=7)

    slider_axis = fig.add_axes([0.25, 0.05, 0.5, 0.025])
    cluster_slider = Slider(
        slider_axis,
        "Superclusters",
        valmin=2,
        valmax=max_clusters,
        valinit=n_superclusters,
        valstep=1,
        valfmt="%.0f",
    )

    def show_spectrum(spectrum, title, samples=None, sample_color="tab:blue"):
        for line in sample_lines:
            line.remove()
        sample_lines.clear()
        if samples is not None:
            sample_lines.extend(plot_bmu_samples(spectrum_axis, bias_mv, samples, rng, sample_color))
        spectrum_line.set_data(bias_mv, spectrum)
        selection_label.set_text(title)
        fig.canvas.draw_idle()

    def update_clusters(value):
        nonlocal supercluster_labels, real_space_labels, label_grid, cmap, norm
        n_clusters = int(value)
        supercluster_labels = cluster_som_nodes(som, n_clusters)
        real_space_labels = supercluster_labels[bmu_ids].reshape(data.shape[:2])
        label_grid = supercluster_labels.reshape(som.map_h, som.map_w)
        cmap = plt.get_cmap("tab20", n_clusters)
        norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)
        som_collection.set_array(label_grid.ravel())
        som_collection.set_cmap(cmap)
        som_collection.set_norm(norm)
        som_collection.set_clim(-0.5, n_clusters - 0.5)
        variance_text.set_text(
            f"Variance Explained\n{variance_explained(real_space_labels.ravel(), n_clusters):.2%}"
        )
        selected_pixels.set_array(np.ma.masked_all(real_space_labels.shape).ravel())
        selected_topography_pixels.set_array(np.ma.masked_all(real_space_labels.shape).ravel())
        data_marker.set_data([], [])
        som_marker.set_data([], [])
        spectrum_line.set_data([], [])
        selection_label.set_text("Click a real-space pixel or SOM node")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return

        if event.inaxes is data_axis:
            column = np.abs(x_nm - event.xdata).argmin()
            row = np.abs(y_nm - event.ydata).argmin()
            data_marker.set_data([x_nm[column]], [y_nm[row]])
            cluster = real_space_labels[row, column]
            show_spectrum(
                data[row, column],
                f"Data spectrum at X={x_nm[column]:.3f} nm, Y={y_nm[row]:.3f} nm (supercluster {cluster})",
            )
        elif event.inaxes is som_axis:
            node_index = np.square(som_coordinates - [event.xdata, event.ydata]).sum(axis=1).argmin()
            row, column = divmod(node_index, som.map_w)
            cluster = label_grid[row, column]
            som_marker.set_data([som_coordinates[node_index, 0]], [som_coordinates[node_index, 1]])
            assigned_pixels = real_space_labels == cluster
            selected_pixels.set_cmap(mcolors.ListedColormap([cmap(cluster)]))
            selected_topography_pixels.set_cmap(mcolors.ListedColormap([cmap(cluster)]))
            selected_pixels.set_array(
                np.ma.masked_where(~assigned_pixels, np.ones_like(assigned_pixels, dtype=float)).ravel()
            )
            selected_topography_pixels.set_array(
                np.ma.masked_where(~assigned_pixels, np.ones_like(assigned_pixels, dtype=float)).ravel()
            )
            cluster_spectra = data.reshape(-1, data.shape[-1])[real_space_labels.ravel() == cluster]
            show_spectrum(
                cluster_spectra.mean(axis=0),
                f"Supercluster {cluster} mean ({len(cluster_spectra)} assigned spectra)",
                cluster_spectra,
                cmap(cluster),
            )
        else:
            return

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    cluster_slider.on_changed(update_clusters)
    plt.show()

def plot_interactive_som_results(topo, data, bias, x, y, som):
    """Explore normalized spectra and SOM prototypes by clicking either map."""
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    bias_mv = bias * 1e3
    x_nm = x * 1e9
    y_nm = y * 1e9
    prototypes = som.weights.detach().cpu().numpy()
    bmus = som.predict(torch.from_numpy(data.reshape(-1, data.shape[-1])))
    bmu_ids = (bmus[:, 0].numpy() * som.map_w + bmus[:, 1].numpy()).reshape(data.shape[:2])
    som_coordinates = som.xy.detach().cpu().numpy()
    spectrum_min, spectrum_max = np.percentile(
        np.concatenate([data.ravel(), prototypes.ravel()]), (1, 99)
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.94, wspace=0.3, hspace=0.3)
    topo_axis, data_axis, som_axis, spectrum_axis = axes.ravel()
    topo_min, topo_max = np.percentile(topo, (1, 99))
    topo_axis.pcolormesh(
        x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max, shading="auto"
    )
    topo_axis.set_title("Topography")
    topo_axis.set_xlabel("X (nm)")
    topo_axis.set_ylabel("Y (nm)")
    topo_axis.set_aspect("equal")
    selected_topography_pixels = topo_axis.pcolormesh(
        x_nm,
        y_nm,
        np.ma.masked_all(zero_bias_slice.shape),
        cmap=mcolors.ListedColormap(["red"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
        shading="auto",
    )
    spectrum_axis.set_xlabel("Bias (mV)")
    spectrum_axis.set_ylabel("Normalized dI/dV")
    spectrum_axis.set_title("Selected Spectrum")
    spectrum_axis.set_xlim(bias_mv.min(), bias_mv.max())
    spectrum_axis.set_ylim(-5, 5)
    spectrum_axis.grid(alpha=0.3)
    spectrum_axis.axvline(0, color="0.25", linewidth=1.25, zorder=0)
    spectrum_line, = spectrum_axis.plot([], [], color="black")
    sample_lines = []
    rng = np.random.default_rng()
    selection_label = spectrum_axis.text(
        0.02, 0.98, "Click a data pixel or SOM node", transform=spectrum_axis.transAxes, va="top"
    )

    data_min, data_max = np.percentile(zero_bias_slice, (1, 99))
    image = data_axis.pcolormesh(
        x_nm,
        y_nm,
        zero_bias_slice,
        cmap="gray",
        vmin=data_min,
        vmax=data_max,
        shading="auto",
    )
    data_axis.set_title(f"Normalized dI/dV at {bias_mv[zero_bias_index]:.1f} mV")
    data_axis.set_xlabel("X (nm)")
    data_axis.set_ylabel("Y (nm)")
    data_axis.set_aspect("equal")
    selected_pixels = data_axis.pcolormesh(
        x_nm,
        y_nm,
        np.ma.masked_all(zero_bias_slice.shape),
        cmap=mcolors.ListedColormap(["red"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
        shading="auto",
    )

    u_matrix = som.get_u_matrix()
    image = plot_som_hexagons(som_axis, som, u_matrix, cmap="viridis")
    som_axis.set_title("SOM U-Matrix")
    som_axis.set_xlabel("SOM column")
    som_axis.set_ylabel("SOM row")

    data_marker, = data_axis.plot([], [], "wo", markeredgecolor="black", markersize=7)
    som_marker, = som_axis.plot([], [], "wo", markeredgecolor="black", markersize=7)

    def show_spectrum(spectrum, title, samples=None):
        for line in sample_lines:
            line.remove()
        sample_lines.clear()
        if samples is not None:
            sample_lines.extend(plot_bmu_samples(spectrum_axis, bias_mv, samples, rng))
        spectrum_line.set_data(bias_mv, spectrum)
        selection_label.set_text(title)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return

        if event.inaxes is data_axis:
            column = np.abs(x_nm - event.xdata).argmin()
            row = np.abs(y_nm - event.ydata).argmin()
            data_marker.set_data([x_nm[column]], [y_nm[row]])
            show_spectrum(
                data[row, column],
                f"Data spectrum at X={x_nm[column]:.3f} nm, Y={y_nm[row]:.3f} nm",
            )
        elif event.inaxes is som_axis:
            node_index = np.square(som_coordinates - [event.xdata, event.ydata]).sum(axis=1).argmin()
            row, column = divmod(node_index, som.map_w)
            som_marker.set_data([som_coordinates[node_index, 0]], [som_coordinates[node_index, 1]])
            node_id = row * som.map_w + column
            assigned_pixels = bmu_ids == node_id
            selected_pixels.set_array(
                np.ma.masked_where(~assigned_pixels, np.ones_like(assigned_pixels, dtype=float)).ravel()
            )
            selected_topography_pixels.set_array(
                np.ma.masked_where(~assigned_pixels, np.ones_like(assigned_pixels, dtype=float)).ravel()
            )
            node_spectra = data.reshape(-1, data.shape[-1])[bmu_ids.ravel() == node_id]
            show_spectrum(
                prototypes[row, column],
                f"SOM prototype at row {row}, column {column} ({assigned_pixels.sum()} assigned spectra)",
                node_spectra,
            )
        else:
            return

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

