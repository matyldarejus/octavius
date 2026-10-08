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
