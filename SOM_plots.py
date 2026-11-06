import torch
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import RadioButtons

from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from SOM_config import CONFIG

ERROR_CMAP = "magma"
ERROR_LIMITS = (0, 100)


def plot_scan_image(axis, x_nm, y_nm, values, **kwargs):
    """Plot scan rows and columns with the transformer's lower-origin convention."""
    values = np.asanyarray(values)
    if values.ndim != 2 or values.shape != (len(y_nm), len(x_nm)):
        raise ValueError("Scan image and coordinate dimensions must agree.")
    if len(x_nm) < 2 or len(y_nm) < 2:
        raise ValueError("Scan coordinates need at least two points per axis.")

    dx = (x_nm[-1] - x_nm[0]) / (len(x_nm) - 1)
    dy = (y_nm[-1] - y_nm[0]) / (len(y_nm) - 1)
    if not np.isfinite((dx, dy)).all() or dx == 0 or dy == 0:
        raise ValueError("Scan coordinate steps must be finite and nonzero.")
    extent = (
        x_nm[0] - dx / 2,
        x_nm[-1] + dx / 2,
        y_nm[0] - dy / 2,
        y_nm[-1] + dy / 2,
    )
    return axis.imshow(
        values, origin="lower", extent=extent, interpolation="nearest", **kwargs
    )


class _InteractiveMapSelection:
    """Keep accumulated map markers, pixel masks, and spectra synchronized."""

    def __init__(self, data_axis, topo_axis, som_axis, spectrum_axis,
                 overlays, x_nm, y_nm, som_coordinates, bias_mv, shape):
        self.figure = data_axis.figure
        self.spectrum_axis = spectrum_axis
        self.overlays = overlays
        self.x_nm, self.y_nm = x_nm, y_nm
        self.som_coordinates = som_coordinates
        self.bias_mv = bias_mv
        self.mask = np.zeros(shape, dtype=bool)
        self.overlay_values = np.ones(shape, dtype=float)
        self.keys = set()
        self.pixels = set()
        self.nodes = set()
        self.lines = []
        self.rng = np.random.default_rng()
        self.spectrum_line, = spectrum_axis.plot([], [], color="black", zorder=3)
        self.label = spectrum_axis.text(
            0.02, 0.98, "", transform=spectrum_axis.transAxes, va="top"
        )
        self.data_markers = [
            axis.plot([], [], "wo", markeredgecolor="black", markersize=7)[0]
            for axis in (data_axis, topo_axis)
        ]
        self.som_marker, = som_axis.plot(
            [], [], "wo", markeredgecolor="black", markersize=7
        )
        self.reset()

    def reset(self):
        self.keys.clear()
        self.pixels.clear()
        self.nodes.clear()
        self.mask.fill(False)
        for line in self.lines:
            line.remove()
        self.lines.clear()
        self.spectrum_line.set_data([], [])
        for marker in (*self.data_markers, self.som_marker):
            marker.set_data([], [])
        for overlay in self.overlays:
            overlay.set_data(np.ma.masked_all(self.mask.shape))
        self.label.set_text("Click a pixel or SOM node; Shift-click to add")

    def select(self, key, spectrum, title, pixel_mask, node_ids, *,
               additive=False, pixel=None, samples=None, color=None):
        if not additive:
            self.reset()
        if key in self.keys:
            return
        self.keys.add(key)
        self.mask |= pixel_mask
        self.nodes.update(node_ids)
        if pixel is not None:
            self.pixels.add(pixel)
        pixels = sorted(self.pixels)
        for marker in self.data_markers:
            marker.set_data(
                [self.x_nm[column] for row, column in pixels],
                [self.y_nm[row] for row, column in pixels],
            )
        coordinates = self.som_coordinates[sorted(self.nodes)]
        self.som_marker.set_data(coordinates[:, 0], coordinates[:, 1])
        for overlay in self.overlays:
            overlay.set_data(np.ma.masked_where(~self.mask, self.overlay_values))
        line_color = color if color is not None else (
            "black" if len(self.keys) == 1 else f"C{(len(self.keys) - 1) % 10}"
        )
        if len(self.keys) == 1:
            self.spectrum_line.set_data(self.bias_mv, spectrum)
            self.spectrum_line.set_color(line_color)
        else:
            self.lines.extend(self.spectrum_axis.plot(
                self.bias_mv, spectrum, color=line_color, zorder=3
            ))
        if samples is not None:
            self.lines.extend(plot_bmu_samples(
                self.spectrum_axis, self.bias_mv, samples, self.rng,
                color if color is not None else "tab:blue",
            ))
        self.label.set_text(
            title if len(self.keys) == 1 else f"{len(self.keys)} selections; last: {title}"
        )
        self.figure.canvas.draw_idle()


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

def _weighted_ward_clusters(som, spectra, bmu_ids, k_min, k_max):
    """Run connected hit-weighted Ward once and record cuts from k_min to k_max."""
    if spectra is None or bmu_ids is None:
        raise ValueError("weighted_ward requires normalized spectra and bmu_ids.")
    n_nodes = som.map_h * som.map_w
    spectra = np.asarray(spectra, dtype=np.float64)
    bmu_ids = np.asarray(bmu_ids)
    if spectra.ndim != 2 or spectra.shape[1] != som.Ne:
        raise ValueError(f"spectra must have shape (n_spectra, {som.Ne}).")
    if not np.isfinite(spectra).all():
        raise ValueError("spectra must contain only finite values.")
    if bmu_ids.shape != (len(spectra),) or not np.issubdtype(bmu_ids.dtype, np.integer):
        raise ValueError("bmu_ids must contain one integer node ID per spectrum.")
    if np.any((bmu_ids < 0) | (bmu_ids >= n_nodes)):
        raise ValueError(f"bmu_ids must be between 0 and {n_nodes - 1}.")

    hits = np.bincount(bmu_ids, minlength=n_nodes)
    sums = np.zeros((n_nodes, som.Ne), dtype=np.float64)
    np.add.at(sums, bmu_ids, spectra)
    prototypes = som.weights.detach().cpu().numpy().reshape(n_nodes, som.Ne)
    means = np.divide(
        sums, hits[:, None], out=prototypes.astype(np.float64, copy=True),
        where=hits[:, None] > 0,
    )
    feature_weights = som.feature_weights.detach().cpu().numpy()
    if not np.isfinite(feature_weights).all() or np.any(feature_weights < 0):
        raise ValueError("weighted_ward requires finite, non-negative feature weights.")
    centroids = means * np.sqrt(feature_weights.astype(np.float64))
    if not np.isfinite(centroids).all():
        raise ValueError("weighted_ward requires finite prototypes for empty nodes.")
    # Empty nodes preserve lattice paths without appreciably shifting hit-weighted means.
    weights = np.where(hits > 0, hits, 1e-6)
    distances = som.D2.detach().cpu().numpy()
    neighbors = (distances > 0) & (distances < 1.5)
    n_components = connected_components(csr_matrix(neighbors), directed=False, return_labels=False)
    if n_components > k_min:
        raise ValueError(
            f"weighted_ward cannot reach {k_min} clusters: "
            f"the hex connectivity graph has {n_components} connected components."
        )

    clusters = {i: (weights[i], centroids[i], [i]) for i in range(n_nodes)}
    adjacency = {i: set(np.flatnonzero(neighbors[i])) for i in range(n_nodes)}
    heap = []

    def push_cost(a, b):
        wa, ca, nodes_a = clusters[a]
        wb, cb, nodes_b = clusters[b]
        if nodes_a[0] > nodes_b[0]:
            a, b = b, a
            nodes_a, nodes_b = nodes_b, nodes_a
        difference = ca - cb
        cost = (wa * wb / (wa + wb)) * np.dot(difference, difference)
        heapq.heappush(heap, (cost, nodes_a[0], nodes_b[0], a, b))

    for a in adjacency:
        for b in sorted(adjacency[a]):
            if a < b:
                push_cost(a, b)

    cuts = {}

    def record_labels():
        labels = np.empty(n_nodes, dtype=int)
        for label, (_, _, nodes) in enumerate(sorted(clusters.values(), key=lambda c: c[2][0])):
            labels[nodes] = label
        cuts[len(clusters)] = labels

    if n_nodes <= k_max:
        record_labels()
    next_id = n_nodes
    while len(clusters) > k_min:
        while heap:
            _, _, _, a, b = heapq.heappop(heap)
            if a in clusters and b in clusters:
                break
        else:
            raise ValueError(
                f"weighted_ward cannot reach {k_min} clusters: "
                "no adjacent cluster pairs remain."
            )
        wa, ca, nodes_a = clusters.pop(a)
        wb, cb, nodes_b = clusters.pop(b)
        weight = wa + wb
        centroid = (wa * ca + wb * cb) / weight
        adjacent = (adjacency.pop(a) | adjacency.pop(b)) - {a, b}
        clusters[next_id] = (weight, centroid, sorted(nodes_a + nodes_b))
        adjacency[next_id] = adjacent
        for neighbor in sorted(adjacent):
            adjacency[neighbor].difference_update((a, b))
            adjacency[neighbor].add(next_id)
            push_cost(next_id, neighbor)
        next_id += 1
        if len(clusters) <= k_max:
            record_labels()
    return cuts


def cluster_som_nodes(som, n_superclusters, linkage=None, spectra=None, bmu_ids=None):
    """Cluster hex-connected nodes using prototypes or hit-weighted spectral means."""
    n_nodes = som.map_h * som.map_w
    if not isinstance(n_superclusters, (int, np.integer)) or not 2 <= n_superclusters <= n_nodes:
        raise ValueError(f"n_superclusters must be between 2 and {n_nodes}")
    linkage = CONFIG.linkage if linkage is None else linkage
    if linkage == "weighted_ward":
        return _weighted_ward_clusters(
            som, spectra, bmu_ids, n_superclusters, n_superclusters
        )[n_superclusters]

    prototypes = som.weights.detach().cpu().numpy().reshape(n_nodes, som.Ne)
    feature_scale = som.feature_weights.detach().cpu().numpy() ** 0.5
    neighbors = (som.D2.detach().cpu().numpy() > 0) & (som.D2.detach().cpu().numpy() < 1.5)
    return AgglomerativeClustering(
        n_clusters=n_superclusters,
        linkage=linkage,
        connectivity=csr_matrix(neighbors),
    ).fit_predict(prototypes * feature_scale)

def get_supercluster_labels(data, som, n_superclusters):
    """Return SOM-node and real-space pixel supercluster IDs."""
    spectra = data.reshape(-1, data.shape[-1])
    bmus = som.predict(torch.from_numpy(spectra))
    bmu_ids = bmus[:, 0].numpy() * som.map_w + bmus[:, 1].numpy()
    supercluster_labels = cluster_som_nodes(
        som, n_superclusters, spectra=spectra, bmu_ids=bmu_ids
    )
    real_space_labels = supercluster_labels[bmu_ids].reshape(data.shape[:2])
    return supercluster_labels, real_space_labels

def supercluster_palette(counts):
    """Return a vivid palette (high saturation and brightness) and cluster-to-size-rank mapping."""
    n_superclusters = len(counts)
    # Only vivid colors so classes stand out against greyscale heatmaps; no dark, muted, or olive tones.
    base_colors = (
        "#FF8C00", "#00BFFF", "#FF1493", "#00E676", "#FFEA00", "#FF1744", "#2979FF",
        "#D500F9", "#00E5FF", "#76FF03", "#FF6E40", "#FF00FF", "#1DE9B6", "#FFC400",
    )
    palette = [mcolors.to_rgb(color) for color in base_colors]
    # Beyond the base set, add golden-ratio-spaced hues at full brightness so colors never turn dull.
    hue = 0.0
    while len(palette) < n_superclusters:
        hue = (hue + 0.618033988749895) % 1.0
        palette.append(tuple(mcolors.hsv_to_rgb((hue, 0.9, 1.0))))
    palette = np.array(palette[:max(n_superclusters, 1)])
    count_order = np.argsort(-counts, kind="stable")
    color_rank = np.empty(n_superclusters, dtype=int)
    color_rank[count_order] = np.arange(n_superclusters)
    return palette, color_rank

def variance_explained_curve(data, som, k_max=15, linkage=None):
    """Return weighted variance explained by superclusters and by all SOM nodes."""
    spectra = data.reshape(-1, data.shape[-1])
    bmus = som.predict(torch.from_numpy(spectra))
    bmu_ids = bmus[:, 0].numpy() * som.map_w + bmus[:, 1].numpy()
    n_nodes = som.map_h * som.map_w
    feature_weights = som.feature_weights.detach().cpu().numpy()
    total_variation = ((spectra - spectra.mean(axis=0)) ** 2 * feature_weights).sum()

    def variance_explained(pixel_labels, n_clusters):
        counts = np.bincount(pixel_labels, minlength=n_clusters)
        sums = np.zeros((n_clusters, spectra.shape[-1]), dtype=spectra.dtype)
        np.add.at(sums, pixel_labels, spectra)
        means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
        residual_variation = ((spectra - means[pixel_labels]) ** 2 * feature_weights).sum()
        return 1 - residual_variation / total_variation if total_variation > 0 else np.nan

    ks = np.arange(2, min(k_max, n_nodes) + 1)
    linkage = CONFIG.linkage if linkage is None else linkage
    cuts = _weighted_ward_clusters(som, spectra, bmu_ids, 2, ks[-1]) if linkage == "weighted_ward" and len(ks) else None
    ve_values = np.array([
        variance_explained(
            (cuts[k] if cuts is not None else cluster_som_nodes(
                som, k, linkage=linkage, spectra=spectra, bmu_ids=bmu_ids
            ))[bmu_ids], k,
        ) for k in ks
    ])
    ceiling = variance_explained(bmu_ids, n_nodes)
    return ks, ve_values, ceiling

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
    fig.suptitle(plot_context["data_name"])
    topo_min, topo_max = np.percentile(topo, (1, 99))
    image = plot_scan_image(axes[0, 0], x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max)
    axes[0, 0].set_title("Topography")
    axes[0, 0].set_xlabel("X (nm)")
    axes[0, 0].set_ylabel("Y (nm)")
    axes[0, 0].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 0], label="Topography")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    data_min, data_max = np.percentile(zero_bias_slice, (1, 99))
    image = plot_scan_image(
        axes[0, 1], x_nm, y_nm, zero_bias_slice,
        cmap="gray", vmin=data_min, vmax=data_max,
    )
    axes[0, 1].set_title(f"Normalized dI/dV at {bias_mv[zero_bias_index]:.1f} mV")
    axes[0, 1].set_xlabel("X (nm)")
    axes[0, 1].set_ylabel("Y (nm)")
    axes[0, 1].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 1], label="Normalized signal")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    plot_scan_image(axes[0, 2], x_nm, y_nm, bmu_ids, cmap=color_key, norm=color_norm)
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
    if plot_context.get("show_plots", True):
        plt.show()
    else:
        plt.close(fig)

def plot_som_superclusters(topo, data, bias, x, y, som, n_superclusters, plot_context):
    """Plot real-space and SOM superclusters with their weighted variance explained."""
    n_nodes = som.map_h * som.map_w
    supercluster_labels, real_space_labels = get_supercluster_labels(data, som, n_superclusters)
    x_nm = x * 1e9
    y_nm = y * 1e9
    label_grid = supercluster_labels.reshape(som.map_h, som.map_w)
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    labels = real_space_labels.ravel()
    counts = np.bincount(labels, minlength=n_superclusters)
    palette, color_rank = supercluster_palette(counts)
    cmap = mcolors.ListedColormap(palette)
    norm = mcolors.BoundaryNorm(np.arange(n_superclusters + 1) - 0.5, n_superclusters)
    real_space_ranks = color_rank[real_space_labels]
    node_ranks = color_rank[label_grid]
    ks, ve_values, ceiling = variance_explained_curve(data, som, k_max=max(15, n_superclusters))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(plot_context["data_name"])
    topo_min, topo_max = np.percentile(topo, (1, 99))
    image = plot_scan_image(axes[0, 0], x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max)
    axes[0, 0].set_title("Topography")
    axes[0, 0].set_xlabel("X (nm)")
    axes[0, 0].set_ylabel("Y (nm)")
    axes[0, 0].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 0], label="Topography")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    didv_min, didv_max = np.percentile(zero_bias_slice, (1, 99))
    image = plot_scan_image(
        axes[0, 1], x_nm, y_nm, zero_bias_slice,
        cmap="gray", vmin=didv_min, vmax=didv_max,
    )
    axes[0, 1].set_title(f"Normalized dI/dV at {bias[zero_bias_index] * 1e3:.1f} mV")
    axes[0, 1].set_xlabel("X (nm)")
    axes[0, 1].set_ylabel("Y (nm)")
    axes[0, 1].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 1], label="Normalized signal")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    image = plot_scan_image(axes[0, 2], x_nm, y_nm, real_space_ranks, cmap=cmap, norm=norm)
    axes[0, 2].set_title("Superclusters (Real Space, Largest to Smallest)")
    axes[0, 2].set_xlabel("X (nm)")
    axes[0, 2].set_ylabel("Y (nm)")
    axes[0, 2].set_aspect("equal")

    axes[1, 0].plot(ks, 100 * ve_values, "-o", markersize=3)
    axes[1, 0].plot(
        [n_superclusters], [100 * ve_values[n_superclusters - 2]], "o",
        color="tab:orange", markersize=9,
    )
    axes[1, 0].axhline(100 * ceiling, color="0.5", linestyle="--", label="all nodes")
    axes[1, 0].set_xlabel("Number of superclusters K")
    axes[1, 0].set_ylabel("Variance explained (%)")
    axes[1, 0].set_title("Variance explained")
    axes[1, 0].set_xticks(ks)
    axes[1, 0].tick_params(axis="x", length=3, width=0.6, color="0.65")
    axes[1, 0].set_axisbelow(True)
    axes[1, 0].grid(axis="x", color="0.8", linewidth=0.5)
    axes[1, 0].legend()

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
        print(f"Saved supercluster plot to {save_path}")
        map_path = os.path.join(
            plot_context["figure_dir"],
            f"{plot_context['save_name']}_{n_superclusters}_segmentation_map.npz",
        )
        np.savez_compressed(
            map_path,
            segmentation_map=palette[real_space_ranks],
            x=x,
            y=y,
        )
        print(f"Saved supercluster segmentation map to {map_path}")
    if plot_context.get("show_plots", True):
        plt.show()
    else:
        plt.close(fig)

    # Plot individual supercluster breakdown (one figure per supercluster)
    plot_individual_superclusters(
        topo, data, bias, x, y, real_space_labels, n_superclusters, palette, color_rank, counts, plot_context
    )

    return supercluster_labels, counts


def plot_individual_superclusters(
    topo,
    data,
    bias,
    x,
    y,
    real_space_labels,
    n_superclusters,
    palette,
    color_rank,
    counts,
    plot_context,
):
    """Plot 3 subfigures for each supercluster: Topo + mask, dI/dV + mask, and sampled + mean spectra."""
    x_nm = x * 1e9
    y_nm = y * 1e9
    bias_mv = bias * 1e3
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    spectra = data.reshape(-1, data.shape[-1])
    flat_labels = real_space_labels.ravel()

    topo_min, topo_max = np.percentile(topo, (1, 99))
    didv_min, didv_max = np.percentile(zero_bias_slice, (1, 99))

    rng = np.random.default_rng(42)

    # Order superclusters from largest to smallest to match the main supercluster overview
    count_order = np.argsort(-counts, kind="stable")

    for rank, cluster_id in enumerate(count_order):
        cluster_mask = real_space_labels == cluster_id
        cluster_color = palette[color_rank[cluster_id]]
        cluster_spectra = spectra[flat_labels == cluster_id]
        n_in_cluster = len(cluster_spectra)
        if n_in_cluster == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        fig.suptitle(
            f"{plot_context['data_name']} — Supercluster {rank + 1}/{n_superclusters} (ID {cluster_id})"
        )

        # 1. Topography with color mask overlay
        plot_scan_image(axes[0], x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max)
        masked_topo = np.ma.masked_where(~cluster_mask, np.ones_like(cluster_mask, dtype=float))
        plot_scan_image(
            axes[0], x_nm, y_nm, masked_topo,
            cmap=mcolors.ListedColormap([cluster_color]),
            vmin=0, vmax=1, alpha=0.55,
        )
        axes[0].set_title(f"Topography (Supercluster {rank + 1}/{n_superclusters})")
        axes[0].set_xlabel("X (nm)")
        axes[0].set_ylabel("Y (nm)")
        axes[0].set_aspect("equal")

        # 2. dI/dV at zero bias with color mask overlay
        plot_scan_image(
            axes[1], x_nm, y_nm, zero_bias_slice,
            cmap="gray", vmin=didv_min, vmax=didv_max,
        )
        masked_didv = np.ma.masked_where(~cluster_mask, np.ones_like(cluster_mask, dtype=float))
        plot_scan_image(
            axes[1], x_nm, y_nm, masked_didv,
            cmap=mcolors.ListedColormap([cluster_color]),
            vmin=0, vmax=1, alpha=0.55,
        )
        axes[1].set_title(f"dI/dV at {bias_mv[zero_bias_index]:.1f} mV")
        axes[1].set_xlabel("X (nm)")
        axes[1].set_ylabel("Y (nm)")
        axes[1].set_aspect("equal")

        # 3. Spectrum plot: ~200 sampled spectra + mean spectrum
        axes[2].set_title(f"Spectra (N={n_in_cluster} points)")
        axes[2].set_xlabel("Bias (mV)")
        axes[2].set_ylabel("Normalized dI/dV")
        axes[2].grid(alpha=0.3)
        axes[2].axvline(0, color="0.25", linewidth=1.25, zorder=0)

        n_samples = min(200, n_in_cluster)
        sample_indices = rng.choice(n_in_cluster, size=n_samples, replace=False)
        sampled_spectra = cluster_spectra[sample_indices]

        axes[2].plot(bias_mv, sampled_spectra.T, color=cluster_color, alpha=0.35, linewidth=0.75)
        mean_spectrum = cluster_spectra.mean(axis=0)
        axes[2].plot(bias_mv, mean_spectrum, color="black", linewidth=2.0, label="Cluster Mean")
        axes[2].legend(loc="upper right")

        save_path = os.path.join(
            plot_context["figure_dir"],
            f"{plot_context['save_name']}_supercluster_{rank + 1}_of_{n_superclusters}_id{cluster_id}.png",
        ) if plot_context["save_images"] else None

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved individual supercluster plot to {save_path}")
        if plot_context.get("show_plots", True):
            plt.show()
        else:
            plt.close(fig)

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
    fig.suptitle(plot_context["data_name"])
    topo_min, topo_max = np.percentile(topo, (1, 99))
    image = plot_scan_image(axes[0, 0], x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max)
    axes[0, 0].set_title("Topography")
    axes[0, 0].set_xlabel("X (nm)")
    axes[0, 0].set_ylabel("Y (nm)")
    axes[0, 0].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 0], label="Topography")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    didv_min, didv_max = np.percentile(zero_bias_slice, (1, 99))
    image = plot_scan_image(
        axes[0, 1], x_nm, y_nm, zero_bias_slice,
        cmap="gray", vmin=didv_min, vmax=didv_max,
    )
    axes[0, 1].set_title(f"Normalized dI/dV at {bias[zero_bias_index] * 1e3:.1f} mV")
    axes[0, 1].set_xlabel("X (nm)")
    axes[0, 1].set_ylabel("Y (nm)")
    axes[0, 1].set_aspect("equal")
    colorbar = fig.colorbar(image, ax=axes[0, 1], label="Normalized signal")
    colorbar.formatter = FormatStrFormatter("%.2f")
    colorbar.update_ticks()

    image = plot_scan_image(
        axes[0, 2], x_nm, y_nm, qe_real_space,
        cmap=ERROR_CMAP, vmin=ERROR_LIMITS[0], vmax=ERROR_LIMITS[1],
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
    if plot_context.get("show_plots", True):
        plt.show()
    else:
        plt.close(fig)




def plot_interactive_som_superclusters(topo, data, bias, x, y, som, n_superclusters, plot_context):
    """Explore real-space spectra and connected supercluster centroids by clicking maps."""
    n_nodes = som.map_h * som.map_w
    max_clusters = min(n_nodes, 20)
    n_superclusters = min(n_superclusters, max_clusters)
    linkage = CONFIG.linkage
    zero_bias_index = np.abs(bias).argmin()
    zero_bias_slice = data[:, :, zero_bias_index]
    bias_mv = bias * 1e3
    spectra = data.reshape(-1, data.shape[-1])
    bmus = som.predict(torch.from_numpy(spectra))
    bmu_ids = bmus[:, 0].numpy() * som.map_w + bmus[:, 1].numpy()
    supercluster_labels = cluster_som_nodes(
        som, n_superclusters, linkage=linkage, spectra=spectra, bmu_ids=bmu_ids
    )
    real_space_labels = supercluster_labels[bmu_ids].reshape(data.shape[:2])
    label_grid = supercluster_labels.reshape(som.map_h, som.map_w)
    som_coordinates = som.xy.detach().cpu().numpy()
    x_nm = x * 1e9
    y_nm = y * 1e9
    ks, ve_values, ceiling = variance_explained_curve(
        data, som, k_max=max_clusters, linkage=linkage
    )

    counts = np.bincount(real_space_labels.ravel(), minlength=n_superclusters)
    palette, color_rank = supercluster_palette(counts)
    cmap = mcolors.ListedColormap(palette)

    norm = mcolors.BoundaryNorm(np.arange(n_superclusters + 1) - 0.5, n_superclusters)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(plot_context["data_name"])
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.15, top=0.94, wspace=0.3, hspace=0.3)
    topo_axis, data_axis, variance_axis, som_axis, spectrum_axis, u_matrix_axis = axes.ravel()
    topo_min, topo_max = np.percentile(topo, (1, 99))
    plot_scan_image(topo_axis, x_nm, y_nm, topo, cmap="gray",
                    vmin=topo_min, vmax=topo_max, alpha=0.45)
    topo_axis.set_title("Topography")
    topo_axis.set_xlabel("X (nm)")
    topo_axis.set_ylabel("Y (nm)")
    topo_axis.set_aspect("equal")
    selected_topography_pixels = plot_scan_image(
        topo_axis, x_nm, y_nm,
        np.ma.masked_all(real_space_labels.shape),
        cmap=mcolors.ListedColormap(["black"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
    )

    data_min, data_max = np.percentile(zero_bias_slice, (1, 99))
    plot_scan_image(
        data_axis, x_nm, y_nm,
        zero_bias_slice,
        cmap="gray",
        vmin=data_min,
        vmax=data_max,
        alpha=0.45,
    )
    data_axis.set_title(f"Normalized dI/dV at {bias_mv[zero_bias_index]:.1f} mV")
    data_axis.set_xlabel("X (nm)")
    data_axis.set_ylabel("Y (nm)")
    data_axis.set_aspect("equal")
    selected_pixels = plot_scan_image(
        data_axis, x_nm, y_nm,
        np.ma.masked_all(real_space_labels.shape),
        cmap=mcolors.ListedColormap(["black"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
    )

    variance_line, = variance_axis.plot(ks, 100 * ve_values, "-o", markersize=3)
    variance_marker, = variance_axis.plot(
        [n_superclusters], [100 * ve_values[n_superclusters - 2]], "o",
        color="tab:orange", markersize=9,
    )
    variance_axis.axhline(100 * ceiling, color="0.5", linestyle="--", label="all nodes")
    variance_axis.set_xlabel("Number of superclusters K")
    variance_axis.set_ylabel("Variance explained (%)")
    variance_axis.set_title("Variance explained")
    variance_axis.set_xticks(ks)
    variance_axis.tick_params(axis="x", length=3, width=0.6, color="0.65")
    variance_axis.set_axisbelow(True)
    variance_axis.grid(axis="x", color="0.8", linewidth=0.5)
    variance_axis.legend()

    som_collection = plot_som_hexagons(som_axis, som, color_rank[label_grid], cmap=cmap, norm=norm,
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
    selection = _InteractiveMapSelection(
        data_axis, topo_axis, som_axis, spectrum_axis,
        (selected_pixels, selected_topography_pixels),
        x_nm, y_nm, som_coordinates, bias_mv, data.shape[:2],
    )
    selection.overlay_values = color_rank[real_space_labels]
    for overlay in selection.overlays:
        overlay.set_cmap(cmap)
        overlay.set_norm(norm)

    variance_axis.set_title("Variance explained (click to select K)")
    linkage_methods = ("ward", "average", "weighted_ward")
    linkage_axis = fig.add_axes([0.76, 0.015, 0.20, 0.10])
    linkage_axis.set_title("Linkage", fontsize=10)
    linkage_selector = RadioButtons(
        linkage_axis, linkage_methods, active=linkage_methods.index(linkage)
    )

    def refresh_clusters():
        nonlocal supercluster_labels, real_space_labels, label_grid, cmap, norm, color_rank
        supercluster_labels = cluster_som_nodes(
            som, n_superclusters, linkage=linkage, spectra=spectra, bmu_ids=bmu_ids
        )
        real_space_labels = supercluster_labels[bmu_ids].reshape(data.shape[:2])
        label_grid = supercluster_labels.reshape(som.map_h, som.map_w)
        counts = np.bincount(real_space_labels.ravel(), minlength=n_superclusters)
        palette, color_rank = supercluster_palette(counts)
        cmap = mcolors.ListedColormap(palette)
        norm = mcolors.BoundaryNorm(np.arange(n_superclusters + 1) - 0.5, n_superclusters)
        som_collection.set_array(color_rank[label_grid].ravel())
        som_collection.set_cmap(cmap)
        som_collection.set_norm(norm)
        som_collection.set_clim(-0.5, n_superclusters - 0.5)
        variance_marker.set_data([n_superclusters], [100 * ve_values[n_superclusters - 2]])
        selection.overlay_values = color_rank[real_space_labels]
        for overlay in selection.overlays:
            overlay.set_cmap(cmap)
            overlay.set_norm(norm)
        selection.reset()
        fig.canvas.draw_idle()

    def update_linkage(value):
        nonlocal linkage, ve_values
        linkage = value
        _, ve_values, _ = variance_explained_curve(
            data, som, k_max=max_clusters, linkage=linkage
        )
        variance_line.set_ydata(100 * ve_values)
        variance_axis.relim()
        variance_axis.autoscale_view()
        refresh_clusters()

    def on_click(event):
        nonlocal n_superclusters
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes is variance_axis:
            if event.button == 1:
                n_clusters = int(ks[np.abs(ks - event.xdata).argmin()])
                if n_clusters != n_superclusters:
                    n_superclusters = n_clusters
                    refresh_clusters()
            return
        additive = event.key is not None and "shift" in event.key.split("+")

        if event.inaxes in (data_axis, topo_axis):
            column = np.abs(x_nm - event.xdata).argmin()
            row = np.abs(y_nm - event.ydata).argmin()
            node_index = bmu_ids[row * data.shape[1] + column]
            cluster = real_space_labels[row, column]
            pixel_mask = np.zeros(data.shape[:2], dtype=bool)
            pixel_mask[row, column] = True
            selection.select(
                ("pixel", row, column),
                data[row, column],
                f"Data spectrum at X={x_nm[column]:.3f} nm, Y={y_nm[row]:.3f} nm (supercluster {cluster})",
                pixel_mask, [node_index], additive=additive, pixel=(row, column),
            )
        elif event.inaxes is som_axis:
            node_index = np.square(som_coordinates - [event.xdata, event.ydata]).sum(axis=1).argmin()
            row, column = divmod(node_index, som.map_w)
            cluster = label_grid[row, column]
            assigned_pixels = real_space_labels == cluster
            cluster_spectra = data.reshape(-1, data.shape[-1])[real_space_labels.ravel() == cluster]
            selection.select(
                ("cluster", cluster),
                cluster_spectra.mean(axis=0),
                f"Supercluster {cluster} mean ({len(cluster_spectra)} assigned spectra)",
                assigned_pixels, [node_index],
                additive=additive, samples=cluster_spectra, color=cmap(color_rank[cluster]),
            )
        else:
            return

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    linkage_selector.on_clicked(update_linkage)
    plt.show()

def plot_interactive_som_results(topo, data, bias, x, y, som, plot_context):
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
    fig.suptitle(plot_context["data_name"])
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.94, wspace=0.3, hspace=0.3)
    topo_axis, data_axis, som_axis, spectrum_axis = axes.ravel()
    topo_min, topo_max = np.percentile(topo, (1, 99))
    plot_scan_image(topo_axis, x_nm, y_nm, topo, cmap="gray", vmin=topo_min, vmax=topo_max)
    topo_axis.set_title("Topography")
    topo_axis.set_xlabel("X (nm)")
    topo_axis.set_ylabel("Y (nm)")
    topo_axis.set_aspect("equal")
    selected_topography_pixels = plot_scan_image(
        topo_axis, x_nm, y_nm,
        np.ma.masked_all(zero_bias_slice.shape),
        cmap=mcolors.ListedColormap(["red"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
    )
    spectrum_axis.set_xlabel("Bias (mV)")
    spectrum_axis.set_ylabel("Normalized dI/dV")
    spectrum_axis.set_title("Selected Spectrum")
    spectrum_axis.set_xlim(bias_mv.min(), bias_mv.max())
    spectrum_axis.set_ylim(-5, 5)
    spectrum_axis.grid(alpha=0.3)
    spectrum_axis.axvline(0, color="0.25", linewidth=1.25, zorder=0)

    data_min, data_max = np.percentile(zero_bias_slice, (1, 99))
    image = plot_scan_image(
        data_axis, x_nm, y_nm,
        zero_bias_slice,
        cmap="gray",
        vmin=data_min,
        vmax=data_max,
    )
    data_axis.set_title(f"Normalized dI/dV at {bias_mv[zero_bias_index]:.1f} mV")
    data_axis.set_xlabel("X (nm)")
    data_axis.set_ylabel("Y (nm)")
    data_axis.set_aspect("equal")
    selected_pixels = plot_scan_image(
        data_axis, x_nm, y_nm,
        np.ma.masked_all(zero_bias_slice.shape),
        cmap=mcolors.ListedColormap(["red"]),
        vmin=0,
        vmax=1,
        alpha=0.8,
    )

    u_matrix = som.get_u_matrix()
    image = plot_som_hexagons(som_axis, som, u_matrix, cmap="viridis")
    som_axis.set_title("SOM U-Matrix")
    som_axis.set_xlabel("SOM column")
    som_axis.set_ylabel("SOM row")

    selection = _InteractiveMapSelection(
        data_axis, topo_axis, som_axis, spectrum_axis,
        (selected_pixels, selected_topography_pixels),
        x_nm, y_nm, som_coordinates, bias_mv, data.shape[:2],
    )

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        additive = event.key is not None and "shift" in event.key.split("+")

        if event.inaxes in (data_axis, topo_axis):
            column = np.abs(x_nm - event.xdata).argmin()
            row = np.abs(y_nm - event.ydata).argmin()
            node_index = bmu_ids[row, column]
            pixel_mask = np.zeros(data.shape[:2], dtype=bool)
            pixel_mask[row, column] = True
            selection.select(
                ("pixel", row, column),
                data[row, column],
                f"Data spectrum at X={x_nm[column]:.3f} nm, Y={y_nm[row]:.3f} nm",
                pixel_mask, [node_index], additive=additive, pixel=(row, column),
            )
        elif event.inaxes is som_axis:
            node_index = np.square(som_coordinates - [event.xdata, event.ydata]).sum(axis=1).argmin()
            row, column = divmod(node_index, som.map_w)
            node_id = row * som.map_w + column
            assigned_pixels = bmu_ids == node_id
            node_spectra = data.reshape(-1, data.shape[-1])[bmu_ids.ravel() == node_id]
            selection.select(
                ("node", node_id),
                prototypes[row, column],
                f"SOM prototype at row {row}, column {column} ({assigned_pixels.sum()} assigned spectra)",
                assigned_pixels, [node_id], additive=additive, samples=node_spectra,
            )
        else:
            return

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
