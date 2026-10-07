"""

Tests the line absorption module.

"""

# default libraries
from dataclasses import replace
from pathlib import Path

# other packages
import h5py
import numpy as np

# testing
import pytest

# internal imports
from octavius.data_management.conventions import OctaviusConfig
from octavius.line_absorption import make_spectra
from octavius.line_absorption.absorption_execution import (
    prepare_sightline_params,
    read_sample_file,
    require_finite_r200c,
)
from octavius.line_absorption.absorption_helpers import SightlineParams, Sightlines
from octavius.line_absorption.sightlines import build_sightlines
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


@pytest.mark.parametrize("overrides", [{"impact_units": "MILES"}, {"absorption_los_axis": "W"}])
def test_invalid_enums_raise(overrides):
    with pytest.raises(ValueError):
        load_config(**overrides)


@pytest.mark.parametrize("field_name", ["n_azimuth", "absorption_chunk_size"])
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


# standalone make_spectra on a pipeline-built test catalogue


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
    )
    return analyse_snapshot(config=config), config


def test_make_spectra_groups_entries_by_axis(absorption_run, tmp_path):
    catalogue_path, config = absorption_run
    with h5py.File(catalogue_path, "r") as f:
        n_galaxies = len(f["galaxy_data/GalID"])
    sample_path = write_sample_file(
        tmp_path / "sample.hdf5", galaxy_idx=[0, 0, n_galaxies - 1], los_axis=["Z", "X", "Z"]
    )

    # KPC units: the junk test haloes may lack a usable r200c
    config = replace(config, absorption_sample_path=sample_path, impact_units="KPC")
    sightline_sets = make_spectra(config=config)

    assert [s.los_axis for s in sightline_sets] == [0, 2]  # X first, then Z
    n_per_entry = (1 if 0.0 in config.impact_parameters else 0) + sum(
        config.n_azimuth for p in config.impact_parameters if p > 0
    )
    assert sightline_sets[0].n_los == 1 * n_per_entry
    assert sightline_sets[1].n_los == 2 * n_per_entry
    assert sorted(set(sightline_sets[1].entry_idx.tolist())) == [0, 2]  # rows of the sample file
    assert set(sightline_sets[1].galaxy_idx.tolist()) == {0, n_galaxies - 1}


def test_make_spectra_requires_a_sample_path(absorption_run):
    _, config = absorption_run
    with pytest.raises(ValueError):
        make_spectra(config=replace(config, absorption_sample_path=None))
