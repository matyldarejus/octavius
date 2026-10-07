"""

Line absorption execution function.

Currently, loads in the galaxy sample file, which requires:
- catalogue galaxy indices,
- (optional) galaxy-specific los axis,
- user-generated bin labels.

Then, builds sightlines:
- relative to galaxies (based on impact parameters + azimuth angles),
- randomly (random nlos across the whole box).

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import OctaviusConfig
    from ..utils.loader import OctaviusCatalogue

# default packages
from pathlib import Path

# other packages
import h5py
import numpy as np
from astropy import units as u

# internal imports
from ..data_management import output_catalogue_path
from ..log import get_logger
from ..utils import load_catalogue
from .absorption_helpers import LOS_AXIS_IDX, SampleEntries, SightlineParams, Sightlines
from .sightlines import build_sightlines

logger = get_logger()


def make_spectra(*, config: OctaviusConfig, catalogue_path: Path | None = None) -> Path:
    """

    Gets a galaxy sample, prepares sightlines, and builds mock absorption spectra
    around a galaxy sample drawn from a catalogue.

    Parameters
    -----------
    config: OctaviusConfig
        Config.
    catalogue_path: pathlib.Path, optional
        Path to an Octavius catalogue.

    Returns
    -----------
    selection_path: pathlib.Path
        Path to the galaxy selection file.

    """
    if config.absorption_sample_path is None:
        raise ValueError("absorption_sample_path is not set in the line_absorption config section.")

    if catalogue_path is None:
        catalogue_path = output_catalogue_path(snapshot_path=config.snapshot_path, output_dir=config.output_dir)
    if not catalogue_path.exists():
        raise FileNotFoundError(f"No Octavius catalogue at {catalogue_path}; run analyse_snapshot first.")

    params = prepare_sightline_params(config=config)

    with load_catalogue(catalogue_path=catalogue_path) as catalogue:
        entries = read_sample_file(
            path=config.absorption_sample_path,
            n_galaxies=len(catalogue.galaxies),
            default_los_axis=LOS_AXIS_IDX[config.absorption_los_axis],
        )
        geometry = _read_galaxy_geometry(catalogue=catalogue, galaxy_idx=entries.galaxy_idx)
        boxsize = catalogue.boxsize_comoving
        scale_factor = float(catalogue.sim_info("scale_factor"))
        hubble_raw = float(catalogue.sim_info("Hz"))
        hubble = (hubble_raw * u.km / u.s / u.Mpc).to(u.km / u.s / u.kpc).value  # km/s/kpc

    if params.impact_in_r200c:
        require_finite_r200c(r200c=geometry["r200c"], galaxy_idx=entries.galaxy_idx)

    sightline_sets: list[Sightlines] = []
    for axis in np.unique(entries.los_axis):  # sort x, y, z in order
        rows = np.flatnonzero(entries.los_axis == axis)  # sample rows projected along this axis
        sightline_sets.append(
            build_sightlines(
                entry_idx=rows,
                galaxy_idx=entries.galaxy_idx[rows],
                centre=geometry["centre"][rows],
                velocity=geometry["velocity"][rows],
                ang_mom=geometry["ang_mom"][rows],
                r200c=geometry["r200c"][rows],
                params=params,
                los_axis=int(axis),
                boxsize=boxsize,
                scale_factor=scale_factor,
                hubble=hubble,
            )
        )

    n_los = sum(s.n_los for s in sightline_sets)
    logger.info(f"Placed {n_los} sightlines for {entries.n_entries} sample entries along {len(sightline_sets)} axes.")
    logger.warning("Gas preselection and spectra are not implemented yet; stopping after sightlines.")  # TODO: C7
    return sightline_sets


def prepare_sightline_params(*, config: OctaviusConfig) -> SightlineParams:
    """

    Parses the sightline config fields into SightlineParams.

    """
    impact_parameters = np.asarray(config.impact_parameters, dtype=np.float64)
    # validate the impact parameters
    if impact_parameters.size == 0 or np.any(~np.isfinite(impact_parameters)) or np.any(impact_parameters < 0.0):
        raise ValueError("impact_parameters must be a non-empty list of finite values >= 0.")

    return SightlineParams(
        impact_parameters=impact_parameters,
        impact_in_r200c=config.impact_units == "R200",
        n_azimuth=config.n_azimuth,
    )


def read_sample_file(*, path: Path, n_galaxies: int, default_los_axis: int) -> SampleEntries:
    """

    Reads and validates the user's galaxy sample file.

    """
    with h5py.File(path, "r") as f:
        if "galaxy_idx" not in f:
            raise KeyError(f"Sample file {path} has no 'galaxy_idx' dataset.")
        galaxy_idx = np.asarray(f["galaxy_idx"][:], dtype=np.int64)
        n_entries = len(galaxy_idx)

        if "los_axis" in f:
            names = [name.upper() for name in f["los_axis"].asstr()[:]]
            unknown = sorted(set(names) - set(LOS_AXIS_IDX))
            if unknown:
                raise ValueError(f"Sample file los_axis has invalid entries {unknown}; use X, Y, or Z.")
            los_axis = np.array([LOS_AXIS_IDX[name] for name in names], dtype=np.int64)
        else:
            los_axis = np.full(n_entries, default_los_axis, dtype=np.int64)

        bin_label = np.asarray(f["bin_label"].asstr()[:]) if "bin_label" in f else None
        attrs = dict(f.attrs)

    # guards for:
    # empty galaxy_idx
    if galaxy_idx.ndim != 1 or n_entries == 0:
        raise ValueError("Sample file galaxy_idx must be a non-empty 1D array.")
    # mismatch length datasets
    if len(los_axis) != n_entries or (bin_label is not None and len(bin_label) != n_entries):
        raise ValueError("Sample file datasets must all have the same length as galaxy_idx.")
    # more galaxies than the catalogue holds
    if np.any((galaxy_idx < 0) | (galaxy_idx >= n_galaxies)):
        raise IndexError(f"Sample file galaxy_idx must lie in [0, {n_galaxies}) (the catalogue's galaxies).")
    # repeated combinations of index + los_axis
    if len(np.unique(np.column_stack((galaxy_idx, los_axis)), axis=0)) != n_entries:
        raise ValueError("Sample file repeats a (galaxy_idx, los_axis) pair.")

    return SampleEntries(galaxy_idx=galaxy_idx, los_axis=los_axis, bin_label=bin_label, attrs=attrs)


def require_finite_r200c(*, r200c: np.ndarray, galaxy_idx: np.ndarray) -> None:
    """

    Flags if sample lacks a finite, positive host r200c.

    """
    bad = ~(np.isfinite(r200c) & (r200c > 0.0))
    if np.any(bad):
        shown = ", ".join(str(g) for g in galaxy_idx[bad][:10])
        raise ValueError(
            f"{int(bad.sum())} sample galaxies have no host r200c (e.g. {shown}); use impact_units: KPC or remove them."
        )


def _read_galaxy_geometry(*, catalogue: OctaviusCatalogue, galaxy_idx: np.ndarray) -> dict[str, np.ndarray]:
    """

    Get the sample galaxy CoM positions, velocities, angular momenta, and host r200c from the catalogue.

    """
    galaxies, haloes = catalogue.galaxies, catalogue.haloes
    host = galaxies.get_membership("field_halo_index")[galaxy_idx]  # -1 if None

    r200c = np.full(len(galaxy_idx), np.nan)
    has_host = host >= 0
    r200c[has_host] = haloes.get_dataset("r200c")[host[has_host]]

    return {
        "centre": galaxies.get_dataset("com_pos_baryon")[galaxy_idx],
        "velocity": galaxies.get_dataset("com_vel_baryon")[galaxy_idx],
        "ang_mom": galaxies.get_dataset("L_baryon")[galaxy_idx],
        "r200c": r200c,
    }
