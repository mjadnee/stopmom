from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SOMConfig:
    file_name: str = "TaS2_055.3ds"
    demo_file_name: str = "demo_stm.npz"
    geometry: str = "rectangular"   # Choose geometry of the network
    weighted: bool = True           # Choose if to weight the vectors by proximity to 0 bias
    zero_bias_width: float = 0.020  # Window of interest, in V
    map_shape: tuple[int, int] | None = None
    seed: int | None = None
    ordering_epochs: int = 20
    tuning_epochs: int = 60
    sigma_end: float = 1.0
    subset_fraction: float = 1.0
    n_superclusters: int | None = None

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return serializable configuration saved with each trained model."""
        return asdict(self)


CONFIG = SOMConfig()