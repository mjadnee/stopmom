from dataclasses import asdict, dataclass

"""
    Required inputs:
        topography: Image with shape (y_pixels, x_pixels).
        data: Spectra with shape (y_pixels, x_pixels, energy_points).
        bias: Bias values with shape (energy_points,).
        x: X coordinates with shape (x_pixels,).
        y: Y coordinates with shape (y_pixels,).

    "data_name" labels the output folder and saved model. 
    Plots are saved as pngs when "save_images" is true; 
    Interactive views open when "interactive_plots" is true;
    "N_superclusters" optionally adds a supercluster plot. 
    Training settings are read from "SOM_config.py".

    Models and figures go under "SOM_outputs/".
"""

@dataclass(frozen=True)
class SOMConfig:
    data_path: str = r"C:\Users\maren\Documents\Yale\daNeto\STM_data"
    data_name: str = "TaS2_055"
    file_type: str= ".3ds"

    save_imgs: bool = False         # Export plots as pngs
    interactive_plots: bool = True  # Trigger interactive plots

    # None skips any supercluster plots.
    # Select an integer value to set the number of superclusters. 
    # This number is adjustable in interactive plots. 
    n_superclusters: int | None = 3
    save_clusters: bool = False      # Export original data and per-pixel supercluster IDs as .npz


    # SOM mapping parameters
    weighted: bool = True           # Choose if to weight the vectors by proximity to 0 bias
    zero_bias_width: float = 0.020  # Window of interest, in Volts
    geometry: str = "rectangular"   # Choose geometry of the network (rectangular, cylindrical, toroidal)
    map_shape: tuple[int, int] | None = None  # CHoose map shape or leave blank. Shape will calculate automatically.
    seed: int = 42

    # Training parameters
    ordering_epochs: int = 20       
    tuning_epochs: int = 60         
    sigma_end: float = 1.0
    subset_fraction: float = 1.0


    def checkpoint_metadata(self) -> dict[str, object]:
        """Return serializable configuration saved with each trained model."""
        return asdict(self)


CONFIG = SOMConfig()

if __name__ == "__main__":
    from SOM_train import main
    main()