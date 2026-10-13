# STM SOM Classifier

## Run the Demo

From the repository directory, run:

demo_data/generate_demo.py

to generate a demo STM dataset npz file. Then, run:

SOM_train.py


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

When `save_clusters` is enabled, the pipeline also writes a compressed `.npz`
file under `SOM_outputs/<data_name>/clusters/`. It contains the original
`topography`, `data`, `bias`, `x`, and `y` arrays, plus `supercluster_ids`, a
2D `(y_pixels, x_pixels)` array assigning a zero-based supercluster ID to each
pixel's spectrum. `n_superclusters` must be set to enable this export.