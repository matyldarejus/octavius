"""

Helper functions for line absorption analysis routines.

Includes shared containers and index maps for the line absorption module.

"""

# default libraries
from dataclasses import dataclass

# other packages
import numpy as np

SSFR_CLASS_IDX: dict[str, int] = {"UNCLASSIFIED": -1, "STAR_FORMING": 0, "GREEN_VALLEY": 1, "QUENCHED": 2}

# classification names in index order
SSFR_CLASS_NAMES: tuple[str, ...] = tuple(
    name for name, idx in sorted(SSFR_CLASS_IDX.items(), key=lambda item: item[1]) if idx >= 0
)


@dataclass(frozen=True, slots=True)
class GalaxyData:
    """

    Galaxy quantities needed for galaxy sampling.
    All arrays have leading dimension n_gal and are in catalogue index order.

    """

    log_mass_star: np.ndarray  # -inf for mass_star <= 0
    sfr: np.ndarray
    log_ssfr: np.ndarray  # LOG_SSFR_FLOOR where sfr <= 0
    r200c: np.ndarray  # r200c of the field halo in kpc, needs confirmation
    is_central: np.ndarray

    @property
    def n_galaxies(self) -> int:
        return len(self.sfr)


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """

    Galaxy sampling parameters read in from the config.

    """

    mode: str  # ALL, RANDOM, BINNED, EXPLICIT
    explicit_indices: np.ndarray  # catalogue indices
    centrals_only: bool
    mass_bin_edges: np.ndarray
    galaxies_per_bin: int
    n_galaxies_random: int
    seed: int
    ssfr_classification: str  # SSFR CUT, MS_OFFSET, NONE
    quenched_definition: str  # SFR_ZERO | BELOW_GREEN_VALLEY
    ssfr_intercept: float  # log10 yr^-1
    ssfr_redshift_slope: float  # dex per unit redshift
    green_valley_width: float  # dex
    ms_slope: float
    ms_intercept: float
    ms_scatter: float  # dex
    sf_n_sigma: float
    gv_n_sigma: float


@dataclass(frozen=True, slots=True)
class GalaxySample:
    """

    Dataclass containing the sampled galaxies.

    Per-galaxy arrays are full catalogue length,
    and mode determines which exactly are selected.

    """

    selected: np.ndarray
    eligible: np.ndarray
    ssfr_class: np.ndarray  # SSFR_CLASS_IDX vals
    bin_idx: np.ndarray  # -1 if ineligble
    bin_labels: tuple[str, ...]
    bin_total: np.ndarray  # eligible population per bin
    mode: str

    @property
    def indices(self) -> np.ndarray:
        # return indices of the selected galaxies
        return np.flatnonzero(self.selected)

    @property
    def n_selected(self) -> int:
        return len(self.indices)
