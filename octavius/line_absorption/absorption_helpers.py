"""

Helper functions for line absorption analysis routines.

Includes shared containers and index maps for the line absorption module.

"""

# default libraries
from collections import namedtuple
from dataclasses import dataclass, field

# other packages
import numpy as np

LOS_AXIS_IDX: dict[str, int] = {"X": 0, "Y": 1, "Z": 2}
KERNEL_IDX = {"CUBIC": 0, "WENDLAND_C2": 2}

SightlineGrid = namedtuple(
    "SightlineGrid",
    [
        "los_x",  # 1st plane axis
        "los_y",  # 2nd plane axis
        "sort_order",  # sightline indices sorted by cell
        "cell_offsets",
        "n_cells_x",
        "n_cells_y",
        "origin_x",  # kpc a
        "origin_y",  # kpc a
        "inv_cell_width",  # 1 / kpc a
    ],
)


# one absorption transition, as read from line_data.csv
Transition = namedtuple("Transition", ["name", "ion", "element", "wavelength", "f", "atomic_weight", "A_ki"])


# per-transition constants for njit
LineConstants = namedtuple(
    "LineConstants",
    [
        "xsec",  # cm^2 km/s: integrated cross-section
        "b_0",  # km/s: thermal Doppler parameter at T = 1 K
        "gamma",  # km/s: natural damping wdith
    ],
)


# element order of the 2D Metallicity dataset (mass fractions); column 0 is total Z
SIMBA_METAL_COLUMNS = ("Z", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe")  # Dave et al. (2019)
SIMBA_C_METAL_COLUMNS = (  # Chem5 (Hough et al. (2023))
    "Z",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
)


def plane_axes(*, los_axis: int) -> tuple[int, int]:
    """

    Right-handed pair of axes spanning the plane perpendicular to the LOS axis.

    """
    return (los_axis + 1) % 3, (los_axis + 2) % 3


@dataclass(frozen=True, slots=True)
class SampleEntries:
    """

    The user's galaxy sample.
    One entry is equivalent to one galaxy along one projection axis.

    """

    galaxy_idx: np.ndarray  # catalogue galaxy indices
    los_axis: np.ndarray  # projection axis desired, if None: defaults to config
    bin_label: np.ndarray  # the user's binning
    attrs: dict = field(default_factory=dict)

    @property
    def n_entries(self) -> int:
        return len(self.galaxy_idx)


@dataclass(frozen=True, slots=True)
class SightlineParams:
    """

    Dataclass containing parameters for generating lines of sight.

    """

    impact_parameters: np.ndarray  # in impact_units, >= 0
    impact_in_r200c: bool  # True: fractions of host r200c; False: physical kpc
    n_azimuth: int


@dataclass(frozen=True, slots=True)
class Sightlines:
    """

    Dataclass containing line of sight properties for the galaxy sample entries.

    These are projected along one axis, through the whole box.

    """

    pos: np.ndarray  # kpc a, in plane_axes order
    entry_idx: np.ndarray
    galaxy_idx: np.ndarray
    impact_param: np.ndarray  # impact parameter in impact_units
    impact: np.ndarray  # comoving impact parameter
    azimuth: np.ndarray  # from the first plane axis
    azimuth_disc: np.ndarray  # from the projected major axis; NaN if face-on or b = 0
    inclination: np.ndarray  # between L and the LOS axis (0 = face-on); NaN if L = 0
    gal_velocity_pos: np.ndarray  # km/s, galaxy LOS velocity incl. Hubble flow
    los_axis: int
    boxsize: float  # kpc a
    vbox: float  # km/s, Hubble velocity across the box

    @property
    def n_los(self) -> int:
        return len(self.galaxy_idx)


@dataclass(frozen=True, slots=True)
class SightlineParticles:
    """

    Gas preselected for one sightline set, in csr format for efficiency.

    """

    particle_idx: np.ndarray
    los_offsets: np.ndarray
    los_members: np.ndarray

    @property
    def n_kept(self) -> int:
        return len(self.particle_idx)


@dataclass(frozen=True, slots=True)
class SightlineSpectra:
    """

    Optical depth along one sightline set for one transition, before any instrument effects.

    """

    line: str  # transition name
    velocity: np.ndarray  # (n_bins,) km/s
    tau: np.ndarray  # (n_los, n_bins)
    weighted: dict[str, np.ndarray]  # quantity to tau-weighted mean (0 where tau = 0)
    member_column: np.ndarray

    @property
    def n_los(self) -> int:
        return self.tau.shape[0]


@dataclass(frozen=True, slots=True)
class GasLayout:
    """

    Stores information about how a simulation stores its gas. Where a field is None, the
    information is read from the snapshot's metadata.

    """

    name: str
    metals_dataset: str  # 2D element mass fractions
    metal_columns: tuple[str, ...]  # element in each columns
    wind_field: str | None  # flag for decoupled wind particles
    support_factor: float | None  # support radius / smoothing length
    baryon_fraction: float | None  # Omega_b / Omega_m of the run
    kernel: str | None  # KERNEL_IDX key

    def column(self, element: str) -> int:
        """

        Returns the metallicity column of an element's mass fraction.

        """
        if element not in self.metal_columns:
            raise KeyError(f"{self.name} does not track '{element}' (tracks: {','.join(self.metal_columns[1:])}).")
        return self.metal_columns.index(element)

    @property
    def stores_hydrogen(self) -> bool:
        """

        True if X_H is its own column; otherwise calculate X_H = Y - Z.

        """
        return "H" in self.metal_columns


GAS_LAYOUTS: dict[str, GasLayout] = {
    "SIMBA": GasLayout(  # Dave et al. (2019)
        name="SIMBA",
        metals_dataset="Metallicity",
        metal_columns=SIMBA_METAL_COLUMNS,
        wind_field="DelayTime",
        support_factor=1.0,
        kernel="CUBIC",
        baryon_fraction=0.048 / 0.3,  # GIZMO SmoothingLength = support radius
    ),
    "SIMBA-C": GasLayout(  # Hough et al. (2023)
        name="SIMBA-C",
        metals_dataset="Metallicity",
        metal_columns=SIMBA_C_METAL_COLUMNS,
        wind_field="DelayTime",
        support_factor=1.0,
        kernel="CUBIC",
        baryon_fraction=0.048 / 0.3,
    ),
    # SWIFT runs: kernel, support and cosmology come from the snapshot; element columns are named in its
    # ElementMassFractions attributes (to be read in the SWIFT step), so only total Z is listed for now
    "SWIFT-KIARA": GasLayout(
        name="SWIFT-KIARA",
        metals_dataset="ElementMassFractions",
        metal_columns=("Z",),
        wind_field="DecouplingDelayTimes",
        support_factor=None,
        kernel=None,
        baryon_fraction=None,
    ),
    "SWIFT-EAGLE": GasLayout(  # feedback does not decouple particles: no winds
        name="SWIFT-EAGLE",
        metals_dataset="ElementMassFractions",
        metal_columns=("Z",),
        wind_field=None,
        support_factor=None,
        kernel=None,
        baryon_fraction=None,
    ),
    "SWIFT-COLIBRE": GasLayout(  # kinetic feedback, but no decoupled winds
        name="SWIFT-COLIBRE",
        metals_dataset="ElementMassFractions",
        metal_columns=("Z",),
        wind_field=None,
        support_factor=None,
        kernel=None,
        baryon_fraction=None,
    ),
    # NOTE: TNG is currently unsupported
}


def detect_gas_layout(*, simulation_type: str, n_metal_columns: int | None) -> GasLayout:
    """

    Returns the gas layout of a snapshot.

    """
    if simulation_type == "SIMBA" and n_metal_columns == 34:
        raise ValueError("Snapshot has 34 Metallicity columns (Simba-C): set simulation_type to SIMBA-C.")
    if simulation_type in GAS_LAYOUTS:
        return GAS_LAYOUTS[simulation_type]
    raise NotImplementedError(f"No gas layout for '{simulation_type}' (have: {', '.join(GAS_LAYOUTS)}).")
