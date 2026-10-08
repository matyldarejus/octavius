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
from ..data_management import OctaviusConstants, build_reader, output_catalogue_path
from ..log import get_logger
from ..utils import load_catalogue
from .absorption_helpers import LOS_AXIS_IDX, SampleEntries, SightlineParams, SightlineParticles, Sightlines
from .sightlines import (
    assemble_sightline_particles,
    build_random_sightlines,
    build_sightline_grid,
    build_sightlines,
    find_overlaps,
)

logger = get_logger()

KERNEL_SUPPORT_FACTOR: dict[str, float] = {  # gamma
    "SIMBA": 1.0,  # GIZMO SmoothingLength is the support radius i.e. H = h
    # "SWIFT-*"; # For SWIFT support radius, H = gamma * h (3.1, https://arxiv.org/html/2305.13380v2)
}
WIND_FIELDS: dict[str, str] = {"SIMBA": "DelayTime"}  # check for any SWIFT fields which are non-zero for wind
SIGHTLINE_CELLS_PER_STRIDE: int = 512  # cell width = boxsize / SIGHTLINE_CELLS_PER_STRIDE


def run_absorption(
    *, config: OctaviusConfig, catalogue_path: Path | None = None
) -> list[tuple[Sightlines, SightlineParticles]]:
    """

    Gets a galaxy sample, prepares sightlines, and builds mock absorption spectra
    around a galaxy sample drawn from a catalogue.

    Parameters
    -----------
    config: OctaviusConfig
        Config.
    catalogue_path: pathlib.Path, optional
        SAMPLE mode: Path to an Octavius catalogue.

    Returns
    -----------
    results: list[tuple[Sightlines, SightlineParticles]]
        One (sightlines, preselected gas) pair per sightline set: per projection axis in SAMPLE mode (X, Y, Z
        order), a single set in RANDOM mode.

    """
    if config.absorption_los_mode == "RANDOM":
        boxsize, scale_factor, hubble = _read_box_from_snapshot(config=config)
        sightline_sets = [
            build_random_sightlines(
                n_los=config.n_random_los,
                los_axis=LOS_AXIS_IDX[config.absorption_los_axis],
                boxsize=boxsize,
                scale_factor=scale_factor,
                hubble=hubble,
                seed=config.absorption_seed,
            )
        ]
    else:
        sightline_sets = _sample_sightlines(config=config, catalogue_path=catalogue_path)

    n_los = sum(s.n_los for s in sightline_sets)
    logger.info(f"Placed {n_los} sightlines ({config.absorption_los_mode} mode) along {len(sightline_sets)} axes.")

    sightline_particles = _preselect_gas(config=config, sightline_sets=sightline_sets)
    for sightlines, particles in zip(sightline_sets, sightline_particles):
        logger.info(
            f"Axis {'XYZ'[sightlines.los_axis]}: {particles.n_kept:,} gas particles kept "
            f"({len(particles.los_members):,} particle-sightline pairs) for {sightlines.n_los} sightlines."
        )

    logger.warning("Spectra are not implemented yet; stopping after gas preselection.")  # TODO: Stage D: Physics

    return list(zip(sightline_sets, sightline_particles))


def _sample_sightlines(
    *, config: OctaviusConfig, catalogue_path: Path | None = None
) -> list[tuple[Sightlines, SightlineParticles]]:
    """

    Sample sightlines based on parameters and galaxy indices provided by the user.

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

    haloes_columns = haloes.keys()  # fix for SIM118 false positive
    if "r200c" in haloes_columns:  # unneccesary after name mismatch is fixed, in-place for testing with test catalogue
        has_host = host >= 0
        r200c[has_host] = haloes.get_dataset("r200c")[host[has_host]]

    return {
        "centre": galaxies.get_dataset("com_pos_baryon")[galaxy_idx],
        "velocity": galaxies.get_dataset("com_vel_baryon")[galaxy_idx],
        "ang_mom": galaxies.get_dataset("L_baryon")[galaxy_idx],
        "r200c": r200c,
    }


def _read_box_from_snapshot(*, config: OctaviusConfig) -> tuple[float, float, float]:
    """

    Reads the box size, the scale factor, and Hubble constant from the snapshot header.

    """
    reader = build_reader(
        snapshot_path=config.snapshot_path, constants=OctaviusConstants(mu=config.MU, frad=config.FRAD), config=config
    )
    sim = reader.simulation_attributes
    hubble = (sim.Hz * u.km / u.s / u.Mpc).to(u.km / u.s / u.kpc).value
    return float(sim.boxsize), float(sim.scale_factor), float(hubble)


def _preselect_gas(*, config: OctaviusConfig, sightline_sets: list[Sightlines]) -> list[SightlineParticles]:
    """

    Streams pos and smoothing_length for ALL gas in contiguous chunks through the reader's serial subset path.

    Each chunk is read once once and tested against every sightline set's grid.

    Experimental and likely deprecated in the future if more efficient method is found.

    """
    sim_type = config.simulation_type
    if sim_type not in KERNEL_SUPPORT_FACTOR:
        raise NotImplementedError(f"Kernel support factor for '{sim_type} not verified yet.")
    support_factor = KERNEL_SUPPORT_FACTOR[sim_type]

    reader = build_reader(
        snapshot_path=config.snapshot_path, constants=OctaviusConstants(mu=config.MU, frad=config.FRAD), config=config
    )
    n_gas = reader.particle_counts["gas"]
    gas_group = reader.inverse_ptype_map["gas"]
    grids = [
        build_sightline_grid(sightlines=s, cell_width=s.boxsize / SIGHTLINE_CELLS_PER_STRIDE) for s in sightline_sets
    ]

    particle_chunks: list[list[np.ndarray]] = [[] for _ in sightline_sets]
    los_chunks: list[list[np.ndarray]] = [[] for _ in sightline_sets]

    # read in the data
    with h5py.File(config.snapshot_path, "r") as snapshot:
        wind_field = WIND_FIELDS.get(sim_type) if config.absorption_exclude_winds else None
        if config.absorption_exclude_winds and wind_field is None:
            logger.warning(f"No wind field known for '{sim_type}'; wind particles will not be excluded.")
        if wind_field is not None and wind_field not in snapshot[gas_group]:
            logger.warning(f"No '{wind_field}' in the snapshot; wind particles will not be excluded.")
            wind_field = None

        for start in range(0, n_gas, config.absorption_chunk_size):  # process chunks, VERY experimental
            stop = min(start + config.absorption_chunk_size, n_gas)
            chunk_idx = np.arange(start, stop, dtype=np.int64)
            columns = reader.read_requested_columns(  # get the position
                ptype="gas", datasets=["pos", "smoothing_length"], sorted_snapshot_indices=chunk_idx
            )
            support = columns["smoothing_length"] * support_factor  # get the support radius
            if wind_field is not None:
                support[snapshot[gas_group][wind_field][start:stop] != 0] = 0.0  # mask out the winds

            for k, (sightlines, grid) in enumerate(zip(sightline_sets, grids)):
                # find particles where the sightline overlaps the particle within its support radius
                rows, los_idx = find_overlaps(
                    part_pos=columns["pos"],
                    support=support,
                    grid=grid,
                    los_axis=sightlines.los_axis,
                    boxsize=sightlines.boxsize,
                )
                particle_chunks[k].append(chunk_idx[rows])
                los_chunks[k].append(los_idx)

            logger.debug("Gas preselection: {stop:,} / {n_gas:,} particles covered.")

    return [
        assemble_sightline_particles(
            particle_idx_chunks=particle_chunks[k], los_idx_chunks=los_chunks[k], n_los=s.n_los
        )
        for k, s in enumerate(sightline_sets)
    ]
