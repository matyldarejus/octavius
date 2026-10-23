"""

Line absorption execution pipeline.


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
from ..data_management.parallel_reading import generate_read_plan
from ..log import get_logger
from ..utils import load_catalogue
from .absorption_computations import (
    build_projected_kernel_table,
    line_of_sight_velocities,
    optical_depth,
    velocity_grid,
)
from .absorption_helpers import (
    KERNEL_IDX,
    LOS_AXIS_IDX,
    GasLayout,
    SampleEntries,
    SightlineParams,
    SightlineParticles,
    Sightlines,
    SightlineSpectra,
    detect_gas_layout,
    plane_axes,
)
from .ion_tables import DEFAULT_BARYON_FRACTION, hydrogen_number_density, ion_counts, ion_fraction, load_ion_table
from .line_profiles import line_constants, load_line_table
from .sightlines import (
    assemble_sightline_particles,
    build_explicit_sightlines,
    build_random_sightlines,
    build_sightline_grid,
    build_sightlines,
    find_overlaps,
)

logger = get_logger()

SIGHTLINE_CELLS_PER_STRIDE: int = 256  # cell width = boxsize / SIGHTLINE_CELLS_PER_STRIDE
GAS_PROPERTIES = ["pos", "vel", "smoothing_length", "rho", "mass", "temperature"]  # properties for second gas read
KPC_CM = (1.0 * u.kpc).to(u.cm).value
MSUN_G = (1.0 * u.M_sun).to(u.g).value
WEIGHTED_QUANTITIES = ("temperature", "rho", "metallicity", "v_pec")  # tau-weighted per pixel


def run_absorption(
    *, config: OctaviusConfig, catalogue_path: Path | None = None
) -> list[tuple[Sightlines, SightlineParticles, dict[str, SightlineSpectra]]]:
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
        One (sightlines, preselected gas, spectra by line name) per sightline set: per projection axis in SAMPLE mode (X, Y, Z
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
    elif config.absorption_los_mode == "EXPLICIT":
        if config.absorption_los_path is None:
            raise ValueError("absorption_los_path is not set in the line_absorption config section.")
        boxsize, scale_factor, hubble = _read_box_from_snapshot(config=config)
        pos, los_axis = read_los_file(
            path=config.absorption_los_path, default_los_axis=LOS_AXIS_IDX[config.absorption_los_axis]
        )
        sightline_sets = []
        for axis in np.unique(los_axis):  # sort x, y, z in order
            rows = np.flatnonzero(los_axis == axis)  # file rows projected along this axis
            sightline_sets.append(
                build_explicit_sightlines(
                    pos=pos[rows],
                    entry_idx=rows,
                    los_axis=int(axis),
                    boxsize=boxsize,
                    scale_factor=scale_factor,
                    hubble=hubble,
                )
            )
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

    spectra = _synthesise_spectra(config=config, sightline_sets=sightline_sets, sightline_particles=sightline_particles)

    return list(zip(sightline_sets, sightline_particles, spectra))


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
        hubble = (float(catalogue.sim_info("Hz")) / u.s).to(u.km / u.s / u.kpc).value  # km/s/kpc

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

        los_axis = _read_los_axes(f=f, n_entries=n_entries, default_los_axis=default_los_axis)

        bin_label = np.asarray(f["bin_label"].asstr()[:]) if "bin_label" in f else None
        attrs = dict(f.attrs)

    # guards for:
    # empty galaxy_idx
    if galaxy_idx.ndim != 1 or n_entries == 0:
        raise ValueError("Sample file galaxy_idx must be a non-empty 1D array.")
    # mismatched length datasets
    if len(los_axis) != n_entries or (bin_label is not None and len(bin_label) != n_entries):
        raise ValueError("Sample file datasets must all have the same length as galaxy_idx.")
    # more galaxies than the catalogue holds
    if np.any((galaxy_idx < 0) | (galaxy_idx >= n_galaxies)):
        raise IndexError(f"Sample file galaxy_idx must lie in [0, {n_galaxies}) (the catalogue's galaxies).")
    # repeated combinations of index + los_axis
    if len(np.unique(np.column_stack((galaxy_idx, los_axis)), axis=0)) != n_entries:
        raise ValueError("Sample file repeats a (galaxy_idx, los_axis) pair.")

    return SampleEntries(galaxy_idx=galaxy_idx, los_axis=los_axis, bin_label=bin_label, attrs=attrs)


def read_los_file(*, path: Path, default_los_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """

    Reads and validates user-provide sightline position file for explicit sampling mode.

    Currently, this is an h5py file as the expectation is that these positions will be chosen
    around specific objects, such that the file will be created in a script.

    This should hold 'pos' (ckpc, (n, 3)) and optional per-row 'los_axis' (X, Y, or Z).

    """
    with h5py.File(path, "r") as f:
        if "pos" not in f:
            raise KeyError(f"Sightline file {path} has no 'pos' dataset.")
        pos = np.asarray(f["pos"][:], dtype=np.float64)

        if "los_axis" in f:
            names = [name.upper() for name in f["los_axis"].asstr()[:]]
            unknown = sorted(set(names) - set(LOS_AXIS_IDX))
            if unknown:
                raise ValueError(f"Sightline file los_axis has invalid entries {unknown}; use X, Y, or Z.")
            los_axis = np.array([LOS_AXIS_IDX[name] for name in names], dtype=np.int64)
        else:
            los_axis = np.full(len(pos), default_los_axis, dtype=np.int64)

    # guards for:
    # wrong shape or empty
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
        raise ValueError(f"Sightline file pos must have shape (n, 3) with n > 0, got {pos.shape}.")
    # non-finite positions
    if not np.all(np.isfinite(pos)):
        raise ValueError("Sightline file pos contains non-finite values.")
    # mismatched length datasets
    if len(los_axis) != len(pos):
        raise ValueError("Sightline file los_axis must have the same length as pos.")

    return pos, los_axis


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
    hubble = (sim.Hz / u.s).to(u.km / u.s / u.kpc).value
    return float(sim.boxsize), float(sim.scale_factor), float(hubble)


def _read_los_axes(*, f: h5py.File, n_entries: int, default_los_axis: int) -> np.ndarray:
    """

    Per-row projection axes from an optional 'los_axis' field, else use the config default.

    """
    if "los_axis" not in f:
        return np.full(n_entries, default_los_axis, dtype=np.int64)
    names = [name.upper() for name in f["los_axis"].asstr()[:]]
    unknown = sorted(set(names) - set(LOS_AXIS_IDX))
    if unknown:
        raise ValueError(f"Sightline file los_axis has invalid entries {unknown}; use X, Y, or Z.")
    return np.array([LOS_AXIS_IDX[name] for name in names], dtype=np.int64)


def _preselect_gas(*, config: OctaviusConfig, sightline_sets: list[Sightlines]) -> list[SightlineParticles]:
    """

    Streams pos and smoothing_length for ALL gas in contiguous chunks through the reader's serial subset path.

    Each chunk is read once once and tested against every sightline set's grid.

    Experimental and likely deprecated in the future if more efficient method is found.

    """
    sim_type = config.simulation_type

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
        layout = _gas_layout(snapshot=snapshot, sim_type=sim_type, gas_group=gas_group)
        support_factor = _kernel_support_factor(snapshot=snapshot, layout=layout)
        wind_field = layout.wind_field if config.absorption_exclude_winds else None
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
                support[snapshot[gas_group][wind_field][start:stop] > 0] = 0.0  # mask out the winds

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

            logger.debug(f"Gas preselection: {stop:,} / {n_gas:,} particles covered.")

    return [
        assemble_sightline_particles(
            particle_idx_chunks=particle_chunks[k], los_idx_chunks=los_chunks[k], n_los=s.n_los
        )
        for k, s in enumerate(sightline_sets)
    ]


def _read_gas_properties(
    *, config: OctaviusConfig, particle_idx: np.ndarray, layout: GasLayout
) -> dict[str, np.ndarray]:
    """

    Reads in the gas properties for the preselected particles only in a second pass.

    These are all properties as listed in GAS_PROPERTIES.

    """
    reader = build_reader(
        snapshot_path=config.snapshot_path, constants=OctaviusConstants(mu=config.MU, frad=config.FRAD), config=config
    )
    gas = reader.read_requested_columns(ptype="gas", datasets=GAS_PROPERTIES, sorted_snapshot_indices=particle_idx)

    with h5py.File(config.snapshot_path, "r") as snapshot:
        metals = snapshot[reader.inverse_ptype_map["gas"]][layout.metals_dataset]
        gas["metals"] = np.empty((len(particle_idx), metals.shape[1]), dtype=np.float64)
        position = 0
        for read_slice, mask in generate_read_plan(idx_sorted=particle_idx):  # contiguous reads
            chunk = metals[read_slice]
            if mask is not None:
                chunk = chunk[mask]
            gas["metals"][position : position + len(chunk)] = chunk
            position += len(chunk)

    return gas


def _synthesise_spectra(
    *, config: OctaviusConfig, sightline_sets: list[Sightlines], sightline_particles: list[SightlineParticles]
) -> list[dict[str, SightlineSpectra]]:
    """

    Computes the optical depth of every requested line along every sightline set.

    Particle properties are read in for all kept particles in preselect_particles. Then, ion
    counts come from the Cloudy tables (with an optional Rahmati self-shielding correction).

    Each set is then synthesised by optical_depth.

    """
    if not config.absorption_lines:
        return [{} for _ in sightline_sets]

    lines = load_line_table()
    unknown = [name for name in config.absorption_lines if name not in lines]
    if unknown:
        raise ValueError(f"Unknown absorption_lines {unknown}; names must match line_absorption/line_data.csv.")

    reader = build_reader(
        snapshot_path=config.snapshot_path, constants=OctaviusConstants(mu=config.MU, frad=config.FRAD), config=config
    )
    sim = reader.simulation_attributes
    redshift, scale_factor = float(sim.redshift), float(sim.scale_factor)
    hubble = (sim.Hz / u.s).to(u.km / u.s / u.kpc).value  # km/s/kpc

    with h5py.File(config.snapshot_path, "r") as snapshot:
        layout = _gas_layout(
            snapshot=snapshot, sim_type=config.simulation_type, gas_group=reader.inverse_ptype_map["gas"]
        )
        support_factor = _kernel_support_factor(snapshot=snapshot, layout=layout)
    if layout.kernel is None:
        raise NotImplementedError(f"Spectra for {layout.name} need its kernel from snapshot metadata.")

    # build the kernel + ion tables
    kernel_table = build_projected_kernel_table(KERNEL_IDX[layout.kernel])
    ion_table = load_ion_table(table_dir=config.ion_table_dir, pattern=config.ion_table_pattern, redshift=redshift)

    logger.info(
        f"Spectra: {', '.join(config.absorption_lines)} || ion tables {config.ion_table_dir} ({config.ion_table_pattern}), "
        f"UVB {config.absorption_uvb}, self-shielding {'on' if config.absorption_self_shielding else 'off'} | "
        f"gas layout {layout.name}, {layout.kernel} kernel."
    )

    # now, do the second gas read
    # get all kept particle indices to read snapshot only once
    kept = np.unique(np.concatenate([p.particle_idx for p in sightline_particles]))

    # get their properties
    gas = _read_gas_properties(config=config, particle_idx=kept, layout=layout)
    gas["support"] = gas["smoothing_length"] * support_factor
    metals = gas["metals"]
    if layout.stores_hydrogen:
        hydrogen_fraction = metals[:, layout.column("H")]
    else:
        hydrogen_fraction = 1.0 - metals[:, layout.column("He")]
    n_H = hydrogen_number_density(density=gas["rho"], hydrogen_fraction=hydrogen_fraction)
    baryon_fraction = layout.baryon_fraction if layout.baryon_fraction is not None else DEFAULT_BARYON_FRACTION

    # find where each sightline set's particle sits in kept particles
    rows = [np.searchsorted(kept, p.particle_idx) for p in sightline_particles]

    # synthesise spectra
    spectra: list[dict[str, SightlineSpectra]] = [{} for _ in sightline_sets]
    for name in config.absorption_lines:
        transition = lines[name]
        constants = line_constants(transition=transition)
        element_fraction = (
            hydrogen_fraction if transition.element == "H" else metals[:, layout.column(transition.element)]
        )
        fraction = ion_fraction(
            table=ion_table,
            ion=transition.ion,
            n_H=n_H,
            T=gas["temperature"],
            redshift=redshift,
            self_shield=config.absorption_self_shielding,
            baryon_fraction=baryon_fraction,
            uvb=config.absorption_uvb,
        )
        gas["n_ion"] = ion_counts(
            fraction=fraction,
            element_mass=element_fraction * gas["mass"] * MSUN_G,
            atomic_weight=transition.atomic_weight,
        )
        gas["b"] = constants.b_0 * np.sqrt(gas["temperature"])

        for k, (sightlines, particles) in enumerate(zip(sightline_sets, sightline_particles)):
            spectra[k][name] = _sightline_set_spectrum(
                name=name,
                sightlines=sightlines,
                particles=particles,
                gas=gas,
                rows=rows[k],
                constants=constants,
                kernel_table=kernel_table,
                scale_factor=scale_factor,
                hubble=hubble,
                pixel_kms=config.absorption_pixel_kms,
            )
        logger.info(f"Synthesised {name} for {sum(s.n_los for s in sightline_sets)} sightlines.")

    return spectra


def _sightline_set_spectrum(
    *, name, sightlines, particles, gas, rows, constants, kernel_table, scale_factor, hubble, pixel_kms
) -> SightlineSpectra:
    """

    Takes one line along one sightline set: the set's particles projected on its plane axes, line centres from the
    Hubble flow and peculiar velocities along its axis, then computes optical_depth.

    """
    los_axis = sightlines.los_axis
    axis_1, axis_2 = plane_axes(los_axis=los_axis)
    pos = gas["pos"][rows]
    v_pec = gas["vel"][rows, los_axis]
    v_line = line_of_sight_velocities(
        x_los=pos[:, los_axis], v_pec_los=v_pec, scale_factor=scale_factor, hubble=hubble, vbox=sightlines.vbox
    )
    # average over each pixel
    weights = np.column_stack([gas["temperature"][rows], gas["rho"][rows], gas["metals"][rows, 0], v_pec])
    n_bins, dv = velocity_grid(vbox=sightlines.vbox, pixel_kms=pixel_kms)  # pixel grid

    tau, weighted, member_column = optical_depth(
        sightlines.pos[:, 0],
        sightlines.pos[:, 1],
        particles.los_offsets,
        particles.los_members,
        np.ascontiguousarray(pos[:, axis_1]),
        np.ascontiguousarray(pos[:, axis_2]),
        gas["support"][rows],
        gas["n_ion"][rows],
        v_line,
        gas["b"][rows],
        weights,
        sightlines.boxsize,
        scale_factor * KPC_CM,
        n_bins,
        dv,
        constants,
        kernel_table,
    )
    return SightlineSpectra(
        line=name,
        velocity=(np.arange(n_bins) + 0.5) * dv,
        tau=tau,
        weighted={quantity: weighted[:, w] for w, quantity in enumerate(WEIGHTED_QUANTITIES)},
        member_column=member_column,
    )


def _gas_layout(*, snapshot: h5py.File, sim_type: str, gas_group: str) -> GasLayout:
    """

    Detects and returns the correct gas layout.

    """
    metals = snapshot[gas_group].get("Metallicity")
    n_columns = metals.shape[1] if metals is not None and metals.ndim == 2 else None
    return detect_gas_layout(simulation_type=sim_type, n_metal_columns=n_columns)


def _kernel_support_factor(*, snapshot: h5py.File, layout: GasLayout) -> float:
    """

    Checks the kernel support radius factor for SWIFT following convention
    in Dehnen & Aly (2012) (doi: 10.1111/j.1365-2966.2012.21439.x).


    SWIFT records this in HydroScheme/"Kernel gamma", where then:

        support radius = kernel gamma * h

    """
    hydro = snapshot.get("HydroScheme")
    if hydro is not None and "Kernel gamma" in hydro.attrs:
        return float(np.atleast_1d(hydro.attrs["Kernel gamma"])[0])  # SWIFT stores this as float32 array
    if layout.support_factor is not None:
        return layout.support_factor
    raise NotImplementedError(f"No kernel support factor for '{layout.name}': not found in its layout or metadata.")
