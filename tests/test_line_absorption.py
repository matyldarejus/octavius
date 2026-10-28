"""

Tests the line absorption module.

"""

# default libraries
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

# other packages
import h5py
import numpy as np

# testing
import pytest
from astropy import units as u
from astropy.constants import codata2022 as codata
from scipy.integrate import quad
from scipy.special import wofz

# internal imports
from octavius.data_management.conventions import OctaviusConfig
from octavius.line_absorption import run_absorption
from octavius.line_absorption.absorption_computations import (
    build_projected_kernel_table,
    kernel_3d,
    optical_depth,
    projected_kernel,
    velocity_grid,
)
from octavius.line_absorption.absorption_execution import (
    _kernel_support_factor,
    _read_gas_properties,
    prepare_sightline_params,
    read_los_file,
    read_sample_file,
    require_finite_r200c,
)
from octavius.line_absorption.absorption_helpers import (
    GAS_LAYOUTS,
    KERNEL_IDX,
    SightlineParams,
    SightlineParticles,
    Sightlines,
    detect_gas_layout,
    plane_axes,
)
from octavius.line_absorption.ion_tables import (
    gamma_hi,
    hydrogen_number_density,
    ion_counts,
    load_ion_table,
    log_ion_fraction,
    read_ion_table,
    self_shielding_factor,
    sigma_hi,
)
from octavius.line_absorption.line_profiles import (
    gaussian,
    gaussian_bin_integral,
    line_constants,
    load_line_table,
    voigt,
    voigt_bin_integral,
)
from octavius.line_absorption.sightlines import (
    assemble_sightline_particles,
    build_explicit_sightlines,
    build_random_sightlines,
    build_sightline_grid,
    build_sightlines,
    find_overlaps,
)
from octavius.line_absorption.spectrum_processing import C_KMS, FWHM_PER_SIGMA, apply_lsf, gaussian_lsf
from octavius.run_octavius import analyse_snapshot
from octavius.utils.generate_snapshots import generate_simba_snapshot

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"
STAGES = {
    "find_galaxies": True,
    "properties_core": True,
    "properties_ptype_specific": True,
    "properties_local_environment": False,
    "photometry": False,
}


# config


def load_config(**overrides) -> OctaviusConfig:
    return OctaviusConfig.from_yaml(config_path=CONFIG_PATH, photometry_table_path=None, **overrides)


def test_line_absorption_section_parses():
    config = load_config()
    assert config.absorption_sample_path == Path("/path/to/sample.hdf5")
    assert config.impact_units == "R200"
    assert config.absorption_los_axis == "Z"


def test_enums_are_uppercased():
    config = load_config(impact_units="kpc", absorption_los_axis="x")
    assert config.impact_units == "KPC"
    assert config.absorption_los_axis == "X"


@pytest.mark.parametrize(
    "overrides", [{"impact_units": "MILES"}, {"absorption_los_axis": "W"}, {"absorption_los_mode": "STOCHASTIC"}]
)
def test_invalid_enums_raise(overrides):
    with pytest.raises(ValueError):
        load_config(**overrides)


@pytest.mark.parametrize("field_name", ["n_azimuth", "absorption_chunk_size", "n_random_los"])
def test_nonpositive_counts_raise(field_name):
    with pytest.raises(ValueError):
        load_config(**{field_name: 0})


@pytest.mark.parametrize("impact_parameters", [[], [-0.5], [np.nan]])
def test_prepare_sightline_params_rejects_bad_impacts(impact_parameters):
    with pytest.raises(ValueError):
        prepare_sightline_params(config=load_config(impact_parameters=impact_parameters))


# sample file


def write_sample_file(path: Path, *, galaxy_idx, los_axis=None, bin_label=None, **attrs) -> Path:
    with h5py.File(path, "w") as f:
        f.create_dataset("galaxy_idx", data=np.asarray(galaxy_idx, dtype=np.int64))
        if los_axis is not None:
            f.create_dataset("los_axis", data=list(los_axis), dtype=h5py.string_dtype())
        if bin_label is not None:
            f.create_dataset("bin_label", data=list(bin_label), dtype=h5py.string_dtype())
        for key, value in attrs.items():
            f.attrs[key] = value
    return path


def test_sample_file_parses_axes_labels_and_attrs(tmp_path):
    path = write_sample_file(
        tmp_path / "s.hdf5", galaxy_idx=[3, 3, 7], los_axis=["z", "X", "Z"], bin_label=["a", "a", "b"], note="test"
    )
    entries = read_sample_file(path=path, n_galaxies=10, default_los_axis=2)
    assert entries.galaxy_idx.tolist() == [3, 3, 7]  # the same galaxy on two axes is allowed
    assert entries.los_axis.tolist() == [2, 0, 2]
    assert entries.bin_label.tolist() == ["a", "a", "b"]
    assert entries.attrs["note"] == "test"


def test_sample_file_default_axis(tmp_path):
    path = write_sample_file(tmp_path / "s.hdf5", galaxy_idx=[1, 2])
    entries = read_sample_file(path=path, n_galaxies=5, default_los_axis=1)
    assert entries.los_axis.tolist() == [1, 1]
    assert entries.bin_label is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"galaxy_idx": [0, 99]},  # out of range
        {"galaxy_idx": [-1]},
        {"galaxy_idx": [2, 2], "los_axis": ["Z", "Z"]},  # repeated (galaxy, axis)
        {"galaxy_idx": [1], "los_axis": ["W"]},  # bad axis
        {"galaxy_idx": [1, 2], "bin_label": ["only one"]},  # length mismatch
        {"galaxy_idx": []},
    ],
)
def test_sample_file_rejects_bad_input(tmp_path, kwargs):
    path = write_sample_file(tmp_path / "s.hdf5", **kwargs)
    with pytest.raises((ValueError, IndexError)):
        read_sample_file(path=path, n_galaxies=10, default_los_axis=2)


def test_require_finite_r200c():
    require_finite_r200c(r200c=np.array([100.0, 50.0]), galaxy_idx=np.array([0, 1]))
    with pytest.raises(ValueError):
        require_finite_r200c(r200c=np.array([100.0, np.nan]), galaxy_idx=np.array([0, 1]))


# los file


def write_los_file(path, *, pos, los_axis=None):
    with h5py.File(path, "w") as f:
        f.create_dataset("pos", data=np.asarray(pos, dtype=np.float64))
        if los_axis is not None:
            f.create_dataset("los_axis", data=np.array(los_axis, dtype="S1"))
    return path


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pos": np.zeros((0, 3))},
        {"pos": np.zeros((2, 2))},
        {"pos": [[np.nan, 0.0, 0.0]]},
        {"pos": np.zeros((2, 3)), "los_axis": ["Z", "W"]},
        {"pos": np.zeros((2, 3)), "los_axis": ["Z"]},
    ],
)
def test_los_file_rejects_bad_input(tmp_path, kwargs):
    path = write_los_file(tmp_path / "los.hdf5", **kwargs)
    with pytest.raises(ValueError):
        read_los_file(path=path, default_los_axis=2)


# sightlines

BOXSIZE = 1000.0  # kpc a
HUBBLE = 0.07  # km/s/kpc (70 km/s/Mpc)


def make_sightline_params(**overrides) -> SightlineParams:
    defaults = SightlineParams(impact_parameters=np.array([0.0, 0.5, 1.0]), impact_in_r200c=True, n_azimuth=4)
    return replace(defaults, **overrides)


def place(*, centre, velocity=None, ang_mom=None, r200c=None, scale_factor=1.0, los_axis=2, **overrides) -> Sightlines:
    centre = np.atleast_2d(np.asarray(centre, dtype=np.float64))
    n = len(centre)
    return build_sightlines(
        entry_idx=np.arange(n) + 100,
        galaxy_idx=np.arange(n) + 10,
        centre=centre,
        velocity=np.zeros((n, 3)) if velocity is None else np.atleast_2d(velocity),
        ang_mom=np.tile([1.0, 0.0, 0.0], (n, 1)) if ang_mom is None else np.atleast_2d(ang_mom),
        r200c=np.full(n, 100.0) if r200c is None else np.asarray(r200c, dtype=np.float64),
        params=make_sightline_params(**overrides),
        los_axis=los_axis,
        boxsize=BOXSIZE,
        scale_factor=scale_factor,
        hubble=HUBBLE,
    )


def periodic_offset(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 0.5 * BOXSIZE) % BOXSIZE - 0.5 * BOXSIZE


def test_sightline_count_and_zero_impact():
    sightlines = place(centre=[[500.0, 500.0, 500.0], [200.0, 200.0, 200.0]])
    assert sightlines.n_los == 2 * (1 + 4 + 4)  # b = 0 collapses to one sightline per entry
    assert sightlines.galaxy_idx.tolist() == [10] * 9 + [11] * 9
    assert sightlines.entry_idx.tolist() == [100] * 9 + [101] * 9
    assert np.allclose(sightlines.pos[0], [500.0, 500.0])  # first sightline: b = 0 through the centre


def test_sightlines_wrap_and_keep_impact_distance():
    sightlines = place(centre=[[990.0, 5.0, 500.0]])  # near the box edge
    assert np.all((sightlines.pos >= 0.0) & (sightlines.pos < BOXSIZE))
    distance = np.linalg.norm(periodic_offset(sightlines.pos, np.array([990.0, 5.0])), axis=1)
    assert np.allclose(distance, sightlines.impact)


def test_physical_kpc_impact_is_converted_to_comoving():
    sightlines = place(centre=[[500.0, 500.0, 500.0]], impact_in_r200c=False, scale_factor=0.5)
    nonzero = sightlines.impact_param > 0
    assert np.allclose(sightlines.impact[nonzero], sightlines.impact_param[nonzero] / 0.5)


def test_plane_axes_follow_los_axis():
    sightlines = place(centre=[[100.0, 200.0, 300.0]], los_axis=0)  # LOS = X: plane = (Y, Z)
    assert np.allclose(sightlines.pos[0], [200.0, 300.0])
    assert sightlines.los_axis == 0


def test_disc_azimuth_and_inclination():
    # L along +X with LOS = Z: edge-on disc, projected major axis along Y
    sightlines = place(centre=[[500.0, 500.0, 500.0]], ang_mom=[[1.0, 0.0, 0.0]])
    phi, phi_disc = sightlines.azimuth, sightlines.azimuth_disc
    has_offset = sightlines.impact > 0
    assert np.isnan(phi_disc[0])  # b = 0
    assert np.allclose(phi_disc[np.isclose(phi, 0.0) & has_offset], np.pi / 2)  # offsets along X = minor axis
    assert np.allclose(phi_disc[np.isclose(phi, np.pi / 2) & has_offset], 0.0)  # offsets along Y = major axis
    assert np.allclose(sightlines.inclination, np.pi / 2)  # edge-on


def test_face_on_disc_has_no_disc_azimuth():
    sightlines = place(centre=[[500.0, 500.0, 500.0]], ang_mom=[[0.0, 0.0, 1.0]])
    assert np.all(np.isnan(sightlines.azimuth_disc))
    assert np.allclose(sightlines.inclination, 0.0)


def test_galaxy_velocity_includes_hubble_flow_and_wraps():
    sightlines = place(centre=[[500.0, 500.0, 400.0]], velocity=[[0.0, 0.0, -50.0]], scale_factor=0.5)
    vbox = HUBBLE * 0.5 * BOXSIZE
    expected = (-50.0 + HUBBLE * 0.5 * 400.0) % vbox
    assert np.allclose(sightlines.gal_velocity_pos, expected)
    assert sightlines.vbox == pytest.approx(vbox)


def random_sightlines(*, n_los=200, seed=0) -> Sightlines:
    return build_random_sightlines(n_los=n_los, los_axis=2, boxsize=BOXSIZE, scale_factor=0.5, hubble=HUBBLE, seed=seed)


def test_random_sightlines_inside_box_with_sentinels():
    sightlines = random_sightlines()
    assert sightlines.n_los == 200
    assert np.all((sightlines.pos >= 0.0) & (sightlines.pos < BOXSIZE))
    assert np.all(sightlines.galaxy_idx == -1) and np.all(sightlines.entry_idx == -1)
    assert np.all(np.isnan(sightlines.gal_velocity_pos)) and np.all(np.isnan(sightlines.impact))
    assert sightlines.vbox == pytest.approx(HUBBLE * 0.5 * BOXSIZE)


def test_random_sightlines_are_seeded():
    assert np.array_equal(random_sightlines(seed=3).pos, random_sightlines(seed=3).pos)
    assert not np.array_equal(random_sightlines(seed=3).pos, random_sightlines(seed=4).pos)


@pytest.mark.parametrize(("los_axis", "expected"), [(2, (1.0, 2.0)), (0, (2.0, 3.0)), (1, (3.0, 1.0))])
def test_explicit_sightlines_use_the_right_handed_plane(los_axis, expected):
    sightlines = build_explicit_sightlines(
        pos=np.array([[1.0, 2.0, 3.0]]),
        entry_idx=np.array([7]),
        los_axis=los_axis,
        boxsize=100.0,
        scale_factor=0.5,
        hubble=0.1,
    )
    np.testing.assert_array_equal(sightlines.pos[0], expected)
    assert sightlines.entry_idx[0] == 7 and sightlines.galaxy_idx[0] == -1
    assert np.isnan(sightlines.impact[0]) and sightlines.vbox == pytest.approx(0.1 * 0.5 * 100.0)


def test_explicit_sightlines_wrap_into_the_box():
    sightlines = build_explicit_sightlines(
        pos=np.array([[-1.0, 102.0, 50.0]]),
        entry_idx=np.array([0]),
        los_axis=2,
        boxsize=100.0,
        scale_factor=1.0,
        hubble=0.1,
    )
    np.testing.assert_allclose(sightlines.pos[0], (99.0, 2.0))


def test_run_absorption_explicit_mode(absorption_run, tmp_path):
    _, config = absorption_run
    path = write_los_file(
        tmp_path / "los.hdf5", pos=[[10.0, 20.0, 30.0], [5.0, 6.0, 7.0], [40.0, 50.0, 60.0]], los_axis=["Z", "X", "Z"]
    )
    results = run_absorption(config=replace(config, absorption_los_mode="EXPLICIT", absorption_los_path=path))

    (x_set, _, x_spectra), (z_set, _, _) = results  # X first, then Z
    assert (x_set.los_axis, z_set.los_axis) == (0, 2)
    np.testing.assert_array_equal(x_set.entry_idx, [1])
    np.testing.assert_allclose(x_set.pos, [[6.0, 7.0]])  # X axis: plane (Y, Z)
    np.testing.assert_array_equal(z_set.entry_idx, [0, 2])
    np.testing.assert_allclose(z_set.pos, [[10.0, 20.0], [40.0, 50.0]])  # Z axis: plane (X, Y)
    assert x_spectra == {}  # absorption_lines=[] in the fixture


def test_run_absorption_explicit_mode_requires_a_path(absorption_run):
    _, config = absorption_run
    with pytest.raises(ValueError, match="absorption_los_path"):
        run_absorption(config=replace(config, absorption_los_mode="EXPLICIT", absorption_los_path=None))


# gas preselection


def brute_force_overlaps(*, part_pos, support, sightlines) -> set[tuple[int, int]]:
    first, second = plane_axes(los_axis=sightlines.los_axis)
    plane_pos = part_pos[:, [first, second]]
    pairs = set()
    for s in range(sightlines.n_los):
        offset = periodic_offset(plane_pos, sightlines.pos[s])
        hits = np.flatnonzero(np.sum(offset**2, axis=1) < support**2)
        pairs.update((int(i), s) for i in hits)
    return pairs


def random_gas(*, n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    part_pos = rng.uniform(0.0, BOXSIZE, size=(n, 3))
    support = rng.uniform(5.0, 150.0, size=n)  # includes kernels that cross the box edge
    return part_pos, support


def grid_overlaps(*, part_pos, support, sightlines, cell_width=25.0) -> tuple[np.ndarray, np.ndarray]:
    grid = build_sightline_grid(sightlines=sightlines, cell_width=cell_width)
    return find_overlaps(part_pos=part_pos, support=support, grid=grid, los_axis=sightlines.los_axis, boxsize=BOXSIZE)


@pytest.mark.parametrize("los_axis", [0, 1, 2])
def test_overlaps_match_brute_force_including_wrap(los_axis):
    sightlines = place(centre=[[500.0, 500.0, 500.0], [985.0, 10.0, 990.0]], los_axis=los_axis)  # one near corners
    part_pos, support = random_gas(n=4000)
    rows, los_idx = grid_overlaps(part_pos=part_pos, support=support, sightlines=sightlines)
    found = set(zip(rows.tolist(), los_idx.tolist()))
    assert len(found) == len(rows)  # no duplicate pairs
    assert found == brute_force_overlaps(part_pos=part_pos, support=support, sightlines=sightlines)


@pytest.mark.parametrize("cell_width", [5.0, 50.0, 500.0])
def test_overlaps_independent_of_cell_width(cell_width):
    sightlines = place(centre=[[300.0, 700.0, 500.0]])
    part_pos, support = random_gas(n=2000)
    rows, los_idx = grid_overlaps(part_pos=part_pos, support=support, sightlines=sightlines, cell_width=cell_width)
    assert set(zip(rows.tolist(), los_idx.tolist())) == brute_force_overlaps(
        part_pos=part_pos, support=support, sightlines=sightlines
    )


def test_overlaps_for_random_sightlines():
    sightlines = random_sightlines(n_los=150)
    part_pos, support = random_gas(n=3000)
    rows, los_idx = grid_overlaps(part_pos=part_pos, support=support, sightlines=sightlines)
    assert set(zip(rows.tolist(), los_idx.tolist())) == brute_force_overlaps(
        part_pos=part_pos, support=support, sightlines=sightlines
    )


def test_zero_support_matches_nothing():
    sightlines = place(centre=[[500.0, 500.0, 500.0]])
    rows, _ = grid_overlaps(part_pos=np.array([[500.0, 500.0, 0.0]]), support=np.array([0.0]), sightlines=sightlines)
    assert len(rows) == 0  # exactly on the central sightline, but masked


def chunked_preselection(*, part_pos, support, sightlines, chunk_size) -> SightlineParticles:
    grid = build_sightline_grid(sightlines=sightlines, cell_width=25.0)
    particle_chunks, los_chunks = [], []
    for start in range(0, len(part_pos), chunk_size):
        idx = np.arange(start, min(start + chunk_size, len(part_pos)))
        rows, los_idx = find_overlaps(
            part_pos=part_pos[idx], support=support[idx], grid=grid, los_axis=sightlines.los_axis, boxsize=BOXSIZE
        )
        particle_chunks.append(idx[rows])
        los_chunks.append(los_idx)
    return assemble_sightline_particles(
        particle_idx_chunks=particle_chunks, los_idx_chunks=los_chunks, n_los=sightlines.n_los
    )


def members(particles: SightlineParticles, s: int) -> set[int]:
    rows = particles.los_members[particles.los_offsets[s] : particles.los_offsets[s + 1]]
    return set(particles.particle_idx[rows].tolist())


def test_chunked_preselection_equals_single_pass():
    sightlines = place(centre=[[500.0, 500.0, 500.0], [985.0, 10.0, 500.0]])
    part_pos, support = random_gas(n=3000)
    single = chunked_preselection(part_pos=part_pos, support=support, sightlines=sightlines, chunk_size=len(part_pos))
    chunked = chunked_preselection(part_pos=part_pos, support=support, sightlines=sightlines, chunk_size=700)
    assert np.array_equal(single.particle_idx, chunked.particle_idx)
    assert np.array_equal(single.los_offsets, chunked.los_offsets)
    assert all(members(single, s) == members(chunked, s) for s in range(sightlines.n_los))


def test_csr_maps_back_to_brute_force():
    sightlines = place(centre=[[500.0, 500.0, 500.0]])
    part_pos, support = random_gas(n=1500)
    particles = chunked_preselection(part_pos=part_pos, support=support, sightlines=sightlines, chunk_size=400)
    reference = brute_force_overlaps(part_pos=part_pos, support=support, sightlines=sightlines)
    rebuilt = {(p, s) for s in range(sightlines.n_los) for p in members(particles, s)}
    assert rebuilt == reference
    assert np.all(np.diff(particles.particle_idx) > 0)


def write_hydro_snapshot(path: Path, **attrs) -> Path:
    with h5py.File(path, "w") as f:
        if attrs:
            group = f.create_group("HydroScheme")
            for key, value in attrs.items():
                group.attrs[key] = value
    return path


def test_kernel_support_factor_read_from_swift_metadata(tmp_path):
    path = write_hydro_snapshot(
        tmp_path / "swift.hdf5",
        **{"Kernel gamma": np.array([1.93649], dtype=np.float32), "Kernel function": "Wendland C2"},
    )
    with h5py.File(path, "r") as snapshot:
        factor = _kernel_support_factor(snapshot=snapshot, layout=GAS_LAYOUTS["SWIFT-KIARA"])
    assert factor == pytest.approx(1.93649, rel=1e-6)


def test_kernel_support_factor_falls_back_to_layout(tmp_path):
    with h5py.File(write_hydro_snapshot(tmp_path / "simba.hdf5"), "r") as snapshot:  # no HydroScheme group
        assert _kernel_support_factor(snapshot=snapshot, layout=GAS_LAYOUTS["SIMBA"]) == 1.0


def test_kernel_support_factor_raises_without_metadata(tmp_path):
    with (
        h5py.File(write_hydro_snapshot(tmp_path / "kiara.hdf5"), "r") as snapshot,
        pytest.raises(NotImplementedError, match="SWIFT-KIARA"),
    ):
        _kernel_support_factor(snapshot=snapshot, layout=GAS_LAYOUTS["SWIFT-KIARA"])  # SWIFT layout, no metadata


# standalone run_absorption on a pipeline-built test catalogue


@pytest.fixture(scope="module")
def absorption_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, OctaviusConfig]:
    tmp_dir = tmp_path_factory.mktemp("absorption")
    snapshot_path = tmp_dir / "test_snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)

    config = OctaviusConfig.from_yaml(  # same "just so it runs" parameters as tests/conftest.py
        config_path=CONFIG_PATH,
        simulation_type="SIMBA",
        snapshot_path=snapshot_path,
        output_dir=tmp_dir,
        cores_per_rank=1,
        halo_id_source="SNAPSHOT",
        subhalo_override=True,
        halo_catalogue_path=None,
        photometry_table_path=None,
        min_dm_per_halo=0,
        min_stars_per_galaxy=2,
        b=1.5,
        velocity_factor=5,
        compress_catalogue=False,
        stages=STAGES,
        absorption_lines=[],
    )
    return analyse_snapshot(config=config), config


def test_run_absorption_groups_entries_by_axis(absorption_run, tmp_path):
    catalogue_path, config = absorption_run
    with h5py.File(catalogue_path, "r") as f:
        n_galaxies = len(f["galaxy_data/GalID"])
    sample_path = write_sample_file(
        tmp_path / "sample.hdf5", galaxy_idx=[0, 0, n_galaxies - 1], los_axis=["Z", "X", "Z"]
    )

    # KPC units: the junk test haloes may lack a usable r200c
    config = replace(config, absorption_sample_path=sample_path, impact_units="KPC")
    results = run_absorption(config=config)

    sightline_sets = [sightlines for sightlines, _, _ in results]

    assert [s.los_axis for s in sightline_sets] == [0, 2]  # X first, then Z
    n_per_entry = (1 if 0.0 in config.impact_parameters else 0) + sum(
        config.n_azimuth for p in config.impact_parameters if p > 0
    )
    assert sightline_sets[0].n_los == 1 * n_per_entry
    assert sightline_sets[1].n_los == 2 * n_per_entry
    assert sorted(set(sightline_sets[1].entry_idx.tolist())) == [0, 2]  # rows of the sample file
    assert set(sightline_sets[1].galaxy_idx.tolist()) == {0, n_galaxies - 1}

    for sightlines, particles, spectra in results:
        assert len(particles.los_offsets) == sightlines.n_los + 1
        assert particles.los_offsets[-1] == len(particles.los_members)
        assert spectra == {}


def test_run_absorption_requires_a_sample_path(absorption_run):
    _, config = absorption_run
    with pytest.raises(ValueError):
        run_absorption(config=replace(config, absorption_sample_path=None))


def test_run_absorption_random_mode_needs_no_sample(absorption_run):
    _, config = absorption_run
    config = replace(config, absorption_los_mode="RANDOM", absorption_sample_path=None, n_random_los=50)
    ((sightlines, particles, _),) = run_absorption(config=config)
    assert sightlines.n_los == 50
    assert sightlines.los_axis == {"X": 0, "Y": 1, "Z": 2}[config.absorption_los_axis]
    assert len(particles.los_offsets) == 51


def test_second_gas_read_matches_snapshot(absorption_run):
    _, config = absorption_run
    with h5py.File(config.snapshot_path, "r") as f:
        gas, h = f["PartType0"], f["Header"].attrs["HubbleParam"]
        n_gas = len(gas["Masses"])
        idx = np.unique(np.r_[0, 1, 2, 3, 40, 45, n_gas // 2, n_gas - 1])  # contiguous run + gaps
        raw_metals, raw_hsml = gas["Metallicity"][idx], gas["SmoothingLength"][idx]

    props = _read_gas_properties(config=config, particle_idx=idx, layout=GAS_LAYOUTS["SIMBA"])

    assert set(props) == {"pos", "vel", "smoothing_length", "rho", "mass", "temperature", "metals"}
    assert all(len(values) == len(idx) for values in props.values())
    np.testing.assert_array_equal(props["metals"], raw_metals)
    np.testing.assert_allclose(props["smoothing_length"], raw_hsml / h, rtol=1e-6)  # kpc a / h -> kpc a
    assert np.all(props["temperature"] > 0.0) and np.all(props["rho"] > 0.0)


def test_run_absorption_synthesises_spectra(absorption_run, tmp_path):
    _, config = absorption_run
    for z in (0.0, 0.5, 1.0):
        write_ion_table(tmp_path / f"lt{round(z * 1e4):05d}f100_i31", z)
    config = replace(
        config,
        absorption_los_mode="RANDOM",
        absorption_sample_path=None,
        n_random_los=40,
        absorption_lines=["OVI1031", "H1215"],
        ion_table_dir=tmp_path,
        ion_table_pattern=TABLE_PATTERN,
    )
    ((sightlines, particles, spectra),) = run_absorption(config=config)

    assert set(spectra) == {"OVI1031", "H1215"}
    for name, spectrum in spectra.items():
        dv = spectrum.velocity[1] - spectrum.velocity[0]
        assert spectrum.tau.shape == (sightlines.n_los, len(spectrum.velocity))
        assert len(spectrum.velocity) * dv == pytest.approx(sightlines.vbox)  # check that pixels tile the box
        assert np.all(spectrum.tau >= 0.0) and len(spectrum.member_column) == len(particles.los_members)
        assert set(spectrum.weighted) == {"temperature", "rho", "metallicity", "v_pec"}

        # every sightline conserves its column: sum_k tau_k dv = xsec * sum of its particles' columns
        cumulative = np.concatenate([[0.0], np.cumsum(spectrum.member_column)])
        column = cumulative[particles.los_offsets[1:]] - cumulative[particles.los_offsets[:-1]]
        xsec = line_constants(transition=load_line_table()[name]).xsec
        assert np.all(spectrum.tau.sum(axis=1) * dv <= xsec * column * (1 + 1e-3))
        has_tau = spectrum.tau > 0.0
        assert np.all(spectrum.weighted["temperature"][has_tau] > 0.0)


def test_unknown_absorption_line_raises(absorption_run):
    _, config = absorption_run
    config = replace(
        config, absorption_los_mode="RANDOM", absorption_sample_path=None, n_random_los=5, absorption_lines=["OVI1067"]
    )
    with pytest.raises(ValueError, match="OVI1067"):
        run_absorption(config=config)


# line data and profiles


def reference_voigt(v, b, gamma):
    """Exact Voigt profile via the Faddeeva function (same definition as pygad)."""
    return np.real(wofz((v + 1j * gamma) / b)) / (b * np.sqrt(np.pi))


def test_line_table_has_every_line_and_known_values():
    lines = load_line_table()
    assert len(lines) == 52  # 49 transitions + Lyman_alpha/beta/gamma aliases
    assert lines["OVI1031"].wavelength == pytest.approx(1031.912)
    assert lines["OVI1031"].element == "O" and lines["OVI1031"].ion == "OVI"
    assert lines["HeII"].A_ki == 0.0  # no damping constant given


def test_line_constants_lyman_alpha():
    constants = line_constants(transition=load_line_table()["H1215"])
    assert constants.b_0 * np.sqrt(1e4) == pytest.approx(12.85, rel=1e-2)  # b(H, 1e4 K) --> 2.85 km/s
    assert constants.xsec == pytest.approx(0.41641 * 0.026540 * 1215.6701e-13, rel=1e-3)  # f * pi e^2/(m_e c) * lambda
    assert constants.gamma == pytest.approx(4.6986e8 * 1215.6701e-13 / (4 * np.pi), rel=1e-6)


@pytest.mark.parametrize("a", [1e-5, 1e-4, 1e-3, 1e-2])
def test_voigt_matches_faddeeva(a):
    b = 10.0
    v = np.linspace(-200.0, 200.0, 4001)
    ours = np.array([voigt(x, b, a * b) for x in v])
    exact = reference_voigt(v, b, a * b)
    assert np.max(np.abs(ours - exact)) < 0.1 * a * exact.max()


def test_voigt_small_x_branch_is_continuous():
    b, gamma = 10.0, 0.01
    x_edge = np.sqrt(1e-4) * b  # where the series branch switches over
    assert voigt(x_edge * (1 - 1e-6), b, gamma) == pytest.approx(voigt(x_edge * (1 + 1e-6), b, gamma), rel=1e-6)


def test_voigt_reduces_to_gaussian():
    assert voigt(3.0, 10.0, 0.0) == pytest.approx(gaussian(3.0, 10.0))


def test_profiles_integrate_to_one():
    b, gamma = 10.0, 0.05
    edges = np.linspace(-500.0, 500.0, 2001)
    gauss_total = sum(gaussian_bin_integral(lo, hi, b) for lo, hi in pairwise(edges))
    voigt_total = sum(voigt_bin_integral(lo, hi, b, gamma, 10) for lo, hi in pairwise(edges))
    assert gauss_total == pytest.approx(1.0, abs=1e-10)
    assert voigt_total == pytest.approx(1.0, abs=1e-3)  # Lorentzian wings beyond +-500 km/s are missing


def test_gaussian_matches_voigt_limit_and_has_unit_area():
    b = 10.0
    v = np.linspace(-100.0, 100.0, 20001)
    phi = np.array([gaussian(x, b) for x in v])
    assert np.trapezoid(phi, v) == pytest.approx(1.0, rel=1e-8)
    assert gaussian(0.0, b) == pytest.approx(1.0 / (b * np.sqrt(np.pi)))
    assert gaussian(b, b) == pytest.approx(np.exp(-1.0) / (b * np.sqrt(np.pi)))


# projected kernels


@pytest.fixture(scope="module", params=list(KERNEL_IDX.values()), ids=list(KERNEL_IDX))
def kernel_table(request):
    return request.param, build_projected_kernel_table(request.param)


@pytest.mark.parametrize("kernel_id", list(KERNEL_IDX.values()))
def test_kernel_3d_has_unit_volume(kernel_id):
    volume, _ = quad(lambda q: 4.0 * np.pi * q * q * kernel_3d(q, kernel_id), 0.0, 1.0, points=[0.5])
    assert volume == pytest.approx(1.0, rel=1e-10)


def test_projected_kernel_has_unit_area(kernel_table):
    _, table = kernel_table
    s = np.linspace(0.0, 1.0, len(table))
    assert np.trapezoid(2.0 * np.pi * s * table, s) == pytest.approx(1.0, rel=1e-5)


def test_projected_kernel_centre_values():
    assert build_projected_kernel_table(KERNEL_IDX["CUBIC"])[0] == pytest.approx(6.0 / np.pi, rel=1e-8)
    assert build_projected_kernel_table(KERNEL_IDX["WENDLAND_C2"])[0] == pytest.approx(7.0 / np.pi, rel=1e-8)


@pytest.mark.parametrize("s", [0.1, 0.3, 0.55, 0.75, 0.9])
def test_projected_kernel_matches_quad(kernel_table, s):
    kernel_id, table = kernel_table
    z_max = np.sqrt(1.0 - s * s)
    points = [np.sqrt(0.25 - s * s)] if s < 0.5 else None  # cubic piece boundary at q = 0.5
    exact, _ = quad(lambda z: 2.0 * kernel_3d(np.sqrt(s * s + z * z), kernel_id), 0.0, z_max, points=points)
    assert projected_kernel(s, table) == pytest.approx(exact, rel=1e-5, abs=1e-7)


def test_projected_kernel_is_monotonic_and_compact(kernel_table):
    _, table = kernel_table
    assert np.all(np.diff(table) <= 0.0)
    assert table[-1] == 0.0
    assert projected_kernel(1.0, table) == 0.0 and projected_kernel(1.5, table) == 0.0


# optical depth / tau synthesis

# set reused variables
KPC_CM = (1.0 * u.kpc).to(u.cm).value
OVI = line_constants(transition=load_line_table()["OVI1031"])
LYA = line_constants(transition=load_line_table()["H1215"])
CUBIC_TABLE = build_projected_kernel_table(KERNEL_IDX["CUBIC"])


def one_particle_spectrum(*, line, column, b, v=500.25, vbox=1000.0, pixel=0.5, s=0.0, weights=None):
    """

    Sets simplest possible system as a test case.
    One particle (H = 1 kpc, a = 1) seen by one sightline; column is its value at s = 0.

    """
    n_bins, dv = velocity_grid(vbox=vbox, pixel_kms=pixel)
    n_ion = column * KPC_CM**2 / CUBIC_TABLE[0]
    w = np.zeros((1, 0)) if weights is None else np.atleast_2d(weights)
    tau, weighted, member_column = optical_depth(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0, 1]),
        np.array([0]),
        np.array([s]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([n_ion]),
        np.array([v]),
        np.array([b]),
        w,
        10_000.0,
        KPC_CM,
        n_bins,
        dv,
        line,
        CUBIC_TABLE,
    )
    v_centres = (np.arange(n_bins) + 0.5) * dv
    return v_centres, dv, tau[0], weighted[0], member_column[0]


def test_tau_integral_equals_xsec_times_column():
    line = OVI._replace(gamma=0.0)
    _, dv, tau, _, column = one_particle_spectrum(line=line, column=1e13, b=20.0)
    assert column == pytest.approx(1e13, rel=1e-12)
    assert tau.sum() * dv == pytest.approx(line.xsec * 1e13, rel=1e-12)


def test_column_follows_projected_kernel():
    _, _, _, _, column = one_particle_spectrum(line=OVI, column=1e13, b=20.0, s=0.5)
    assert column == pytest.approx(1e13 * projected_kernel(0.5, CUBIC_TABLE) / CUBIC_TABLE[0], rel=1e-12)


def test_line_centre_and_width():
    line = OVI._replace(gamma=0.0)
    v, dv, tau, _, _ = one_particle_spectrum(line=line, column=1e13, b=20.0, pixel=0.1)
    mean = np.sum(v * tau) / tau.sum()
    var = np.sum((v - mean) ** 2 * tau) / tau.sum()
    assert mean == pytest.approx(500.25, abs=1e-9)
    # b^2 / 2 plus Sheppard's correction for binned data
    assert var == pytest.approx(20.0**2 / 2 + dv**2 / 12, rel=1e-6)


def test_line_centre_optical_depth_matches_standard_formula():
    t = load_line_table()["OVI1031"]
    line = OVI._replace(gamma=0.0)
    _, _, tau, _, _ = one_particle_spectrum(line=line, column=1e13, b=20.0, v=500.025, pixel=0.05)
    tau_0 = 1.497e-15 * t.f * t.wavelength * 1e13 / 20.0  # Morton (2003)
    assert tau.max() == pytest.approx(tau_0, rel=1e-3)


def test_periodic_wrap_is_symmetric_and_translation_invariant():
    line = OVI._replace(gamma=0.0)
    _, dv, tau_mid, _, _ = one_particle_spectrum(line=line, column=1e13, b=20.0, v=500.25)
    _, _, tau_edge, _, _ = one_particle_spectrum(line=line, column=1e13, b=20.0, v=0.25)  # centre of pixel 0
    np.testing.assert_allclose(tau_edge[1:], tau_edge[:0:-1], rtol=1e-12)  # symmetric across the wrap
    np.testing.assert_allclose(tau_edge, np.roll(tau_mid, -round(500.0 / dv)), rtol=1e-12, atol=1e-30)
    assert tau_edge.sum() == pytest.approx(tau_mid.sum(), rel=1e-12)


def test_damping_wing_truncation_loss():
    column, b = 1e15, 20.0
    v, dv, tau, _, _ = one_particle_spectrum(line=LYA, column=column, b=b, vbox=10_000.0)
    half_width = np.sqrt(LYA.xsec * column * LYA.gamma / (np.pi * 1e-6))
    deficit = 1.0 - tau.sum() * dv / (LYA.xsec * column)
    offset = np.mod(v - 500.25 + 5000.0, 10_000.0) - 5000.0  # nearest image
    assert deficit == pytest.approx(2.0 * LYA.gamma / (np.pi * half_width), abs=1e-4)
    assert np.all(tau[np.abs(offset) > half_width + dv] == 0.0)


def test_minimum_image_in_plane():
    n_bins, dv = velocity_grid(vbox=1000.0, pixel_kms=0.5)
    args = (np.array([0, 1]), np.array([0]))
    _, _, col_wrapped = optical_depth(
        np.array([0.0]),
        np.array([0.0]),
        *args,
        np.array([10_000.0 - 0.3]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1e60]),
        np.array([500.25]),
        np.array([20.0]),
        np.zeros((1, 0)),
        10_000.0,
        KPC_CM,
        n_bins,
        dv,
        OVI,
        CUBIC_TABLE,
    )
    _, _, col_direct = optical_depth(
        np.array([0.0]),
        np.array([0.0]),
        *args,
        np.array([0.3]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1e60]),
        np.array([500.25]),
        np.array([20.0]),
        np.zeros((1, 0)),
        10_000.0,
        KPC_CM,
        n_bins,
        dv,
        OVI,
        CUBIC_TABLE,
    )
    assert col_wrapped[0] == pytest.approx(col_direct[0], rel=1e-9) and col_direct[0] > 0.0


def test_tau_weighted_extras():
    _, _, tau, weighted, _ = one_particle_spectrum(line=OVI, column=1e13, b=20.0, weights=[[1e4, 0.5]])
    has_tau = tau > 0.0
    np.testing.assert_allclose(weighted[0, has_tau], 1e4, rtol=1e-12)
    np.testing.assert_allclose(weighted[1, has_tau], 0.5, rtol=1e-12)


@pytest.mark.parametrize("log_column", [13.0, 16.0, 19.0])
def test_curve_of_growth_matches_faddeeva(log_column):
    column, b, vbox, v_line = 10.0**log_column, 20.0, 10_000.0, 5000.0
    v, dv, tau, _, _ = one_particle_spectrum(line=LYA, column=column, b=b, v=v_line, vbox=vbox, pixel=1.0)
    offset = np.mod(v - v_line + 0.5 * vbox, vbox) - 0.5 * vbox  # nearest image, as in the synthesis
    tau_ref = LYA.xsec * column * reference_voigt(offset, b, LYA.gamma)
    ew, ew_ref = np.sum(1.0 - np.exp(-tau)) * dv, np.sum(1.0 - np.exp(-tau_ref)) * dv
    assert ew == pytest.approx(ew_ref, rel=1e-3)


@pytest.mark.parametrize("pixel", [0.25, 1.0, 2.5])
def test_equivalent_width_converges_with_pixel_size(pixel):
    line = OVI._replace(gamma=0.0)
    _, dv_fine, tau_fine, _, _ = one_particle_spectrum(line=line, column=1e15, b=20.0, pixel=0.05)
    _, dv, tau, _, _ = one_particle_spectrum(line=line, column=1e15, b=20.0, pixel=pixel)
    ew_fine = np.sum(1.0 - np.exp(-tau_fine)) * dv_fine
    assert np.sum(1.0 - np.exp(-tau)) * dv == pytest.approx(ew_fine, rel=1e-2)


# ion tables

# setup test constants
ION_HEADER = "HydrogenI HeliumII  CarbonIV  OxygenVI  OxygeVIII MagnesiII SilicoIII IronII"  # truncated as in i31
ION_NAMES = ("HI", "HeII", "CIV", "OVI", "OVIII", "MgII", "SiIII", "FeII")
N_IONS = len(ION_NAMES)
LOG_NH_GRID = np.arange(-9.0, 1.01, 0.5)
LOG_T_GRID = np.arange(1.0, 9.01, 0.25)
TABLE_PATTERN = "lt<z>f100_i31"


def synthetic_log_fraction(log_nH, log_T, k, z):
    """

    Exactly bilinear in (log n_H, log T) and linear in z, so interpolation should hopefully reproduce it.

    """
    return -1.0 - 0.1 * (k + 1) * log_nH - 0.05 * log_T + 0.01 * log_nH * log_T + 0.3 * z


def write_ion_table(path, z, header=ION_HEADER, n_ions=N_IONS):
    """

    Make a test ion table file.

    """
    nH, T = np.meshgrid(LOG_NH_GRID, LOG_T_GRID, indexing="ij")  # T fastest, as in the Cloudy files
    columns = [nH.ravel(), T.ravel()] + [synthetic_log_fraction(nH, T, k, z).ravel() for k in range(n_ions)]
    with path.open("w") as f:
        f.write(f"#Hdens  Temp    {header} redshift= {z:.5f}\n")
        np.savetxt(f, np.column_stack(columns), fmt="%.12f")


# part 1 : check functions responsible for reading and interpolating the tables ------------------
@pytest.fixture
def ion_table_dir(tmp_path):
    for z in (0.0, 0.5, 1.0):
        write_ion_table(tmp_path / f"lt{round(z * 1e4):05d}f100_i31", z)  # pygad's naming: digits = z without the point
    (tmp_path / "README").write_text("ignored: does not match the pattern")
    return tmp_path


def test_ion_table_header_names_are_correct(ion_table_dir):
    table = read_ion_table(path=ion_table_dir / "lt05000f100_i31")
    assert table.ions == ION_NAMES and table.redshift == 0.5
    assert table.log_fraction.shape == (len(LOG_NH_GRID), len(LOG_T_GRID), len(ION_NAMES))


def test_ion_table_header_with_missing_space(tmp_path):
    path = tmp_path / "lt00000f100_i45"
    write_ion_table(path, 0.0, header="SulphurIV SulphurVSulphurVI IronII", n_ions=4)
    assert read_ion_table(path=path).ions == ("SIV", "SV", "SVI", "FeII")


def test_ion_table_rejects_header_column_mismatch(tmp_path):
    path = tmp_path / "lt00000f100_i31"
    write_ion_table(path, 0.0, n_ions=len(ION_NAMES) - 1)
    with pytest.raises(ValueError, match="data columns"):
        read_ion_table(path=path)


def test_ion_fraction_exact_at_grid_points(ion_table_dir):
    """

    Test for wrong interpolation weights.

    """
    table = load_ion_table(table_dir=ion_table_dir, pattern=TABLE_PATTERN, redshift=0.5)
    nH, T = np.meshgrid(LOG_NH_GRID, LOG_T_GRID, indexing="ij")
    got = log_ion_fraction(table=table, ion="OVI", log_nH=nH.ravel(), log_T=T.ravel())
    np.testing.assert_allclose(got, synthetic_log_fraction(nH, T, 3, 0.5).ravel(), atol=1e-10)


def test_ion_fraction_bilinear_between_grid_points(ion_table_dir):
    table = load_ion_table(table_dir=ion_table_dir, pattern=TABLE_PATTERN, redshift=0.5)
    rng = np.random.default_rng(1)
    log_nH, log_T = rng.uniform(-9.0, 1.0, 1000), rng.uniform(1.0, 9.0, 1000)
    got = log_ion_fraction(table=table, ion="OVI", log_nH=log_nH, log_T=log_T)
    np.testing.assert_allclose(got, synthetic_log_fraction(log_nH, log_T, 3, 0.5), atol=1e-10)


def test_ion_fraction_clamps_outside_grid(ion_table_dir):
    table = load_ion_table(table_dir=ion_table_dir, pattern=TABLE_PATTERN, redshift=0.5)
    got = log_ion_fraction(table=table, ion="OVI", log_nH=np.array([-12.0, 3.0]), log_T=np.array([0.0, 10.0]))
    expected = synthetic_log_fraction(np.array([-9.0, 1.0]), np.array([1.0, 9.0]), 3, 0.5)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_redshift_interpolation_and_edges(ion_table_dir):
    def load(z):  # would use lambda but ruff doesn't like it
        return load_ion_table(table_dir=ion_table_dir, pattern=TABLE_PATTERN, redshift=z)

    z0, z05, z1 = (read_ion_table(path=ion_table_dir / f"lt{n}f100_i31") for n in ("00000", "05000", "10000"))
    np.testing.assert_allclose(load(0.25).log_fraction, 0.5 * (z0.log_fraction + z05.log_fraction), atol=1e-12)
    np.testing.assert_array_equal(load(0.5).log_fraction, z05.log_fraction)
    np.testing.assert_array_equal(load(2.0).log_fraction, z1.log_fraction)  # beyond the tables: edge table


def test_unknown_ion_raises(ion_table_dir):
    table = load_ion_table(table_dir=ion_table_dir, pattern=TABLE_PATTERN, redshift=0.0)
    with pytest.raises(KeyError, match="NII"):
        log_ion_fraction(table=table, ion="NII", log_nH=np.zeros(1), log_T=np.full(1, 4.0))


def test_mismatched_tables_raise(tmp_path):
    write_ion_table(tmp_path / "lt00000f100_i31", 0.0)
    write_ion_table(tmp_path / "lt05000f100_i31", 0.5, header="HydrogenI HeliumII", n_ions=2)
    with pytest.raises(ValueError, match="different grids or ions"):
        load_ion_table(table_dir=tmp_path, pattern=TABLE_PATTERN, redshift=0.25)


# part 2 : check functions responsible for dealing with UV background and self-sheilding ----------
@pytest.mark.parametrize(
    ("uvb", "gamma_0", "sigma_0"),
    [("HM12", 2.25e-14, 2.67e-18), ("FG20", 4.694e-14, -4.89946e-19 + 3.072648e-18)],  # pygad tables / sigHI fits
)
def test_uvb_rates_at_z0(uvb, gamma_0, sigma_0):
    assert gamma_hi(redshift=0.0, uvb=uvb) == pytest.approx(gamma_0)
    assert sigma_hi(redshift=0.0, uvb=uvb) == pytest.approx(sigma_0, rel=1e-12)


def test_unknown_uvb_raises():
    with pytest.raises(ValueError, match="P19"):
        sigma_hi(redshift=0.0, uvb="P19")


def test_hydrogen_density_and_ion_counts():
    assert hydrogen_number_density(density=np.array([codata.m_p.cgs.value]), hydrogen_fraction=0.75)[
        0
    ] == pytest.approx(0.75)
    oxygen_mass = 1e10 * 15.999 * codata.u.cgs.value  # grams of 1e10 oxygen atoms
    assert ion_counts(fraction=np.array([0.2]), element_mass=np.array([oxygen_mass]), atomic_weight=15.999)[
        0
    ] == pytest.approx(2e9)


@pytest.mark.parametrize(
    ("uvb", "gamma_0", "sigma_0"),
    [("HM12", 2.25e-14, 2.67e-18), ("FG20", 4.694e-14, -4.89946e-19 + 3.072648e-18)],  # pygad tables / sigHI fits
)
def test_self_shielding_limits_and_characteristic_density(uvb, gamma_0, sigma_0):
    """

    Check the Rahmati factor for 1) very thin gas (f ~ 1),
    2) x = n_H / n_H,ss = 1,
    3) very dense gas (f ~ 1e-3)
    4) hot gas (f for n_H,ss prop to T^0.17)

    """
    n_ss = 6.73e-3 * (sigma_0 / 2.49e-18) ** (-2 / 3) * (0.16 / 0.17) ** (-1 / 3) * (gamma_0 * 1e12) ** (2 / 3)
    f_at_n_ss = 0.98 * 2.0**-2.28 + 0.02 * 2.0**-0.84  # x = n_H / n_H,ss = 1
    f = self_shielding_factor(n_H=np.array([1e-10, n_ss, 1e3]), T=np.full(3, 1e4), redshift=0.0, uvb=uvb)
    assert f[0] == pytest.approx(1.0, abs=1e-5)  # optically thin
    assert f[1] == pytest.approx(f_at_n_ss, rel=1e-12)
    assert f[2] == 1e-3  # floored in dense gas
    hot = self_shielding_factor(n_H=np.array([n_ss * 10.0**0.17]), T=np.array([1e5]), redshift=0.0, uvb=uvb)
    assert hot[0] == pytest.approx(f_at_n_ss, rel=1e-12)  # n_H,ss scales as T^0.17


# gas layouts


def test_unknown_layouts_and_untracked_elements_raise():
    with pytest.raises(NotImplementedError, match="TNG"):
        detect_gas_layout(simulation_type="TNG", n_metal_columns=None)
    with pytest.raises(KeyError, match="Al"):
        GAS_LAYOUTS["SIMBA"].column("Al")


@pytest.mark.parametrize("name", ["SIMBA", "SIMBA-C"])
def test_every_line_element_is_available(name):
    layout = GAS_LAYOUTS[name]
    for element in {t.element for t in load_line_table().values()}:
        if element == "H" and not layout.stores_hydrogen:
            continue  # Simba: X_H = 1 - Y - Z, no column needed
        layout.column(element)  # KeyError if a line's element is missing


# spectrum processing


@pytest.mark.parametrize("resolving_power", [20_000, 40_000, 50_000])
def test_gaussian_lsf_width_matches_resolving_power(resolving_power):
    dv = 2.5
    kernel = gaussian_lsf(resolving_power=resolving_power, dv=dv)
    v = (np.arange(len(kernel)) - len(kernel) // 2) * dv
    sigma = C_KMS / resolving_power / FWHM_PER_SIGMA
    assert kernel.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.sum(kernel * v) == pytest.approx(0.0, abs=1e-12)  # centred on the middle pixel
    assert np.sum(kernel * v**2) == pytest.approx(sigma**2 + dv**2 / 12, rel=1e-4)  # pixels add dv^2 / 12


def test_apply_lsf_conserves_equivalent_width():
    v = np.arange(400) * 2.5
    flux = np.exp(-2.0 * np.exp(-(((v - 5.0) / 10.0) ** 2)))[None, :]  # line at the box edge, so its LSF wraps
    smoothed = apply_lsf(flux=flux, kernel=gaussian_lsf(resolving_power=20_000, dv=2.5))
    np.testing.assert_allclose(np.sum(1.0 - smoothed), np.sum(1.0 - flux), rtol=1e-12)
    assert smoothed.min() > flux.min()  # the line gets shallower


def test_apply_lsf_turns_one_absorbed_pixel_into_the_kernel():
    kernel = gaussian_lsf(resolving_power=40_000, dv=2.5)
    half = len(kernel) // 2
    flux = np.ones((1, 100))
    flux[0, 0] = 0.0  # absorbed pixel at the box edge
    expected = np.zeros(100)
    expected[np.arange(-half, half + 1) % 100] = kernel
    np.testing.assert_allclose(1.0 - apply_lsf(flux=flux, kernel=kernel)[0], expected, atol=1e-15)


def test_apply_lsf_rejects_kernel_wider_than_spectrum():
    kernel = gaussian_lsf(resolving_power=20_000, dv=2.5)
    with pytest.raises(ValueError, match="wider than the spectrum"):
        apply_lsf(flux=np.ones((1, len(kernel) - 1)), kernel=kernel)
