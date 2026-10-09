# STM SOM Classifier

## Run the Demo

From the repository directory, run:

```powershell
python SOM_train.py
```

This loads `demo_data/demo_stm.npz`, a small synthetic dataset, and trains a
SOM. The generated checkpoint is written under the ignored `SOM_outputs/`
directory. Run this once to check that the environment and training path work.

## Use Your Own Data

`SOM_train.py` is independent of any STM file format. Load or preprocess your
data with your preferred tools, then pass the five required NumPy arrays to
`train_som`:

```python
from SOM_train import train_som

som = train_som(
    topography=topography,
    data=spectra,
    bias=bias,
    x=x,
    y=y,
    data_name="my_dataset",
)
```

The required data cube and axes are:

| Array | Shape | Description |
| --- | --- | --- |
| `data` | `(y_pixels, x_pixels, energy_points)` | Conductance spectrum at each spatial pixel. |
| `topography` | `(y_pixels, x_pixels)` | Topographic image aligned with `data`. |
| `bias` | `(energy_points,)` | Bias voltage for each spectral point. |
| `x` | `(x_pixels,)` | X coordinate of each column. |
| `y` | `(y_pixels,)` | Y coordinate of each row. |

`train_som` z-score normalizes each spectrum before training and returns the
trained `SOM`. Set `save_images=True` to save diagnostic figures, or
`interactive_plots=True` to show interactive figures. Models and figures are
written under `SOM_outputs/`, which is ignored by Git.

`SOM_train_local.py` is reserved for machine-specific data loading and is also
ignored by Git.