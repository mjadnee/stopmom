# STM SOM Classifier

## Run the Demo

From the repository directory, run:

demo_data/generate_demo.py

to generate a demo STM dataset npz file. Then, run:

SOM_train.py

Toroidal SOMs use a flat-torus initialization in four principal components
(at least four energy points are required). Automatic grid sizing uses the
variance ratio of PC1+PC2 to PC3+PC4 and keeps an even row count.
Checkpoint names continue to use `toroidal`. Existing matching checkpoints are
reloaded as usual; remove an older checkpoint to retrain with the new initialization.
Rectangular and cylindrical models are unchanged.

## Use Your Own Data

Load or preprocess your data with your preferred tools. 
The required data cube and axes are:

| Array | Shape | Description |
| --- | --- | --- |
| `data` | `(y_pixels, x_pixels, energy_points)` | Conductance spectrum at each spatial pixel. |
| `topography` | `(y_pixels, x_pixels)` | Topographic image aligned with `data`. |
| `bias` | `(energy_points,)` | Bias voltage for each spectral point. |
| `x` | `(x_pixels,)` | X coordinate of each column. |
| `y` | `(y_pixels,)` | Y coordinate of each row. |


You can change parameters and interact with what plots are generated
(options for static or interactive plots) and if plots are saved as pngs
via the SOM_config file. 

- `show_plots`: show static plot windows when `True`; suppress them when `False`.
- `save_imgs`: save static plots as PNGs and a standalone supercluster RGB map
  as `.npz`, even when `show_plots` is `False`.
- `interactive_plots`: open interactive plots independently of `show_plots`.
- `linkage`: supercluster merge method (`ward`, `average`, or `weighted_ward`), shared by static
  plots, interactive plots, and exports.

Ward splits spectra into compact, balanced groups. Average follows U-matrix
boundaries. Weighted Ward minimises pixel-level variance using each node's hit
count and the mean of its assigned normalized spectra, rather than its
neighbourhood-smoothed SOM prototype. All three use the SOM feature weights
and permit merges only between adjacent hex-lattice clusters. Weighted Ward
favours the majority of pixels: low-hit and empty nodes are absorbed easily,
and small rare regions are less likely to get their own cluster. Empty nodes
keep their prototypes with a tiny weight (`1e-6`) to preserve lattice connectivity.
The clustering API requires normalized spectra and their flattened BMU IDs
when `weighted_ward` is selected; plotting and exports supply these automatically.

Set `show_plots = False` and `interactive_plots = True` to open only interactive
plots. With `show_plots = True` and `save_imgs = True`, static plots are both
saved and displayed.

In either interactive view, click a pixel in the dI/dV or topography image
to display its spectrum and place a white, black-outlined marker on its
best-matching SOM prototype (BMU). In the supercluster view, the marker
identifies the exact prototype within the colored supercluster, not its centroid.
Hold Shift while clicking to add pixels or SOM selections, including across
panels. Selected pixel spectra and SOM prototypes are overlaid in the spectrum
panel; supercluster selections overlay each cluster's mean spectrum and sampled
spectra. Both scan images highlight the union of selected pixels and the pixels
assigned to selected nodes or superclusters. SOM markers identify selected nodes
and pixel BMUs (one marker at the clicked node per selected supercluster). Repeated Shift-clicks
on the same item do not duplicate it. A click without Shift replaces the entire
selection with the clicked item.

In the interactive supercluster plot, click the variance-explained graph to
update the number of superclusters. Clicks snap to the nearest available integer
K, from 2 through 20 (or the number of SOM nodes, if smaller).
The curve highlights the selected value. Changing K clears the current
pixel and spectrum selection; clicking the already selected K leaves it intact.
Static and interactive variance-explained plots show integer K ticks with
short grey tick marks and faint grey vertical guide lines.
Use the Linkage radio buttons to switch between `ward`, `average`, and `weighted_ward`
without reopening the plot. The initial selection comes from
`linkage` in the configuration. Switching methods updates the clusters and
variance-explained curve and clears the previous spectrum selection. This
changes only the current interactive view, not the configuration or exports.

Static and interactive supercluster plots share a vivid palette (no black, grey,
or muted tones, so classes stay visible over greyscale heatmaps), assigned from
largest to smallest cluster by pixel count. Selection overlays and sampled
spectra use the same cluster colors. Beyond the 14 base colors, additional
full-brightness hues are generated automatically.

With `save_imgs = True` and `n_superclusters` set, the map-only export is saved
as `<save_name>_<linkage>_<n_superclusters>_segmentation_map.npz` under
`SOM_outputs/<data_name>/figures/`. It contains `segmentation_map`, an RGB array
of shape `(y_pixels, x_pixels, 3)` with floating-point values in [0, 1], and
the original `x` and `y` coordinate arrays in their input units. Colors match
the static supercluster plot. Rows and columns align directly with `topography`
and `data`, without flipping, transposing, or resizing. There are no axes,
titles, legends, or spectra in this export; `save_clusters` is not required.

```python
import numpy as np

with np.load("path_to_segmentation_map.npz", allow_pickle=False) as saved:
    segmentation_map = saved["segmentation_map"]
    x, y = saved["x"], saved["y"]

# To display with the same orientation as the classifier:
# plt.imshow(segmentation_map, origin="lower", interpolation="nearest")
```

When `save_clusters` is enabled, the pipeline also writes a compressed `.npz`
file under `SOM_outputs/<data_name>/clusters/`. It contains the original
`topography`, `data`, `bias`, `x`, and `y` arrays, plus `supercluster_ids`, a
2D `(y_pixels, x_pixels)` array assigning a zero-based supercluster ID to each
pixel's spectrum. It also stores the plot colors for each ID:
`supercluster_colors`, an `(n_superclusters, 3)` RGB array in [0, 1], and
`supercluster_color_hex`, an `(n_superclusters,)` array of hex strings, where
row `i` is the color of supercluster ID `i`. `n_superclusters` must be set to
enable this export.

Figure and cluster export filenames include the linkage name so runs with
different methods do not overwrite each other. Cluster exports are named
`<save_name>_<linkage>_<n_superclusters>_superclusters.npz`. The SOM model
checkpoint name and training are unchanged by the linkage option.