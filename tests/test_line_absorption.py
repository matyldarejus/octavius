"""

Tests the line absorption module.

Right now, includes: config parsing, galaxy sampling, the pipeline stage and the standalone
galaxy selection.

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
from octavius.line_absorption.absorption_execution import _find_central_galaxies, prepare_sampling_params
from octavius.line_absorption.absorption_helpers import SSFR_CLASS_IDX, GalaxyData, SamplingParams
from octavius.line_absorption.galaxy_sampling import (
    LOG_SSFR_FLOOR,
    assign_bins,
    classify_ssfr,
    compute_log_ssfr,
    select_galaxies,
)
from octavius.run_octavius import analyse_snapshot
from octavius.utils.generate_snapshots import generate_simba_snapshot

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"
STAGES = {
    "find_galaxies": True,
    "properties_core": True,
    "properties_ptype_specific": True,
    "properties_local_environment": False,
    "photometry": False,
    "line_absorption": True,
}

SF = SSFR_CLASS_IDX["STAR_FORMING"]
GV = SSFR_CLASS_IDX["GREEN_VALLEY"]
Q = SSFR_CLASS_IDX["QUENCHED"]
UNC = SSFR_CLASS_IDX["UNCLASSIFIED"]


# config


def load_config(**overrides) -> OctaviusConfig:
    return OctaviusConfig.from_yaml(config_path=CONFIG_PATH, photometry_table_path=None, **overrides)


def test_line_absorption_section_parses():
    config = load_config()
    assert config.stages["line_absorption"] is False
    assert config.galaxy_selection == "BINNED"
    assert config.mass_bin_edges == [10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5]
    assert config.explicit_galaxy_indices == []


def test_enums_are_uppercased():
    config = load_config(galaxy_selection="binned", ssfr_classification="ms_offset")
    assert config.galaxy_selection == "BINNED"
    assert config.ssfr_classification == "MS_OFFSET"


@pytest.mark.parametrize(
    "overrides",
    [{"galaxy_selection": "SOMETIMES"}, {"ssfr_classification": "VIBES"}, {"quenched_definition": "MAYBE"}],
)
def test_invalid_enums_raise(overrides):
    with pytest.raises(ValueError):
        load_config(**overrides)


@pytest.mark.parametrize("field_name", ["galaxies_per_bin", "n_galaxies_random"])
def test_nonpositive_counts_raise(field_name):
    with pytest.raises(ValueError):
        load_config(**{field_name: 0})


@pytest.mark.parametrize("edges", [[10.0], [10.0, 10.0, 11.0], [11.0, 10.0]])
def test_prepare_sampling_params_rejects_bad_edges(edges):
    config = load_config(mass_bin_edges=edges)
    with pytest.raises(ValueError):
        prepare_sampling_params(config=config)


# galaxy sampling


DEFAULT_PARAMS = SamplingParams(
    mode="BINNED",
    explicit_indices=np.array([], dtype=np.int64),
    centrals_only=True,
    mass_bin_edges=np.array([10.0, 10.5, 11.0]),
    galaxies_per_bin=2,
    n_galaxies_random=3,
    seed=42,
    ssfr_classification="SSFR_CUT",
    quenched_definition="SFR_ZERO",
    ssfr_intercept=-10.8,
    ssfr_redshift_slope=0.3,
    green_valley_width=1.0,
    ms_slope=0.73,
    ms_intercept=-7.33,
    ms_scatter=0.39,
    sf_n_sigma=1.0,
    gv_n_sigma=3.0,
)


def make_params(**overrides) -> SamplingParams:
    return replace(DEFAULT_PARAMS, **overrides)


def make_galaxies(*, log_mass_star, log_ssfr, sfr=None, r200c=None, is_central=None) -> GalaxyData:
    n = len(log_mass_star)
    return GalaxyData(
        log_mass_star=np.asarray(log_mass_star, dtype=np.float64),
        sfr=np.ones(n) if sfr is None else np.asarray(sfr, dtype=np.float64),
        log_ssfr=np.asarray(log_ssfr, dtype=np.float64),
        r200c=np.full(n, 100.0) if r200c is None else np.asarray(r200c, dtype=np.float64),
        is_central=np.ones(n, dtype=bool) if is_central is None else np.asarray(is_central, dtype=bool),
    )


def make_population(*, n: int, seed: int = 0) -> GalaxyData:
    rng = np.random.default_rng(seed)
    return make_galaxies(log_mass_star=rng.uniform(10.0, 11.5, n), log_ssfr=rng.uniform(-11.0, -9.0, n))


def test_compute_log_ssfr_floors_zero_sfr():
    log_ssfr = compute_log_ssfr(sfr=np.array([0.0, 1.0]), mass_star=np.array([1e10, 1e10]))
    assert log_ssfr[0] == LOG_SSFR_FLOOR
    assert log_ssfr[1] == pytest.approx(-10.0)


@pytest.mark.parametrize(
    "quenched_definition, expected",
    [
        ("SFR_ZERO", [SF, SF, GV, UNC, Q]),
        ("BELOW_GREEN_VALLEY", [SF, SF, GV, Q, Q]),
    ],
)
def test_classify_ssfr_thresholds(quenched_definition, expected):
    # at z = 0: sf_threshold = -10.8 (inclusive), gv_floor = -11.8 (inclusive)
    galaxies = make_galaxies(
        log_mass_star=[10.2] * 5,
        log_ssfr=[-9.0, -10.8, -11.8 + 1e-9, -11.8 - 1e-9, LOG_SSFR_FLOOR],
        sfr=[1.0, 1.0, 1.0, 1.0, 0.0],
    )
    params = make_params(quenched_definition=quenched_definition)
    assert classify_ssfr(galaxy_data=galaxies, redshift=0.0, params=params).tolist() == expected


def test_classify_ssfr_nan_sfr_is_unclassified():
    galaxies = make_galaxies(log_mass_star=[10.2], log_ssfr=[LOG_SSFR_FLOOR], sfr=[np.nan])
    params = make_params(quenched_definition="BELOW_GREEN_VALLEY")
    assert classify_ssfr(galaxy_data=galaxies, redshift=0.0, params=params).tolist() == [UNC]


def test_none_classification_bins_by_mass_only():
    galaxies = make_population(n=50)
    sample = select_galaxies(galaxy_data=galaxies, params=make_params(ssfr_classification="NONE"), redshift=0.0)
    assert np.all(sample.ssfr_class == UNC)
    assert len(sample.bin_labels) == 2


def test_mass_bin_edges():
    log_mass_star = np.array([10.0, 10.5, 11.0, 9.99, 11.01])
    bin_idx, _ = assign_bins(
        log_mass_star=log_mass_star,
        ssfr_class=np.full(5, UNC),
        eligible=np.ones(5, dtype=bool),
        mass_bin_edges=np.array([10.0, 10.5, 11.0]),
        bin_by_ssfr=False,
    )
    assert bin_idx.tolist() == [0, 1, 1, -1, -1]  # lower edge inclusive, last bin closed


def test_criteria_exclude_satellites_and_missing_r200c():
    galaxies = make_galaxies(
        log_mass_star=[10.2, 10.2, 10.2],
        log_ssfr=[-10.0, -10.0, -10.0],
        r200c=[100.0, np.nan, 100.0],
        is_central=[True, True, False],
    )
    sample = select_galaxies(galaxy_data=galaxies, params=make_params(mode="ALL"), redshift=0.0)
    assert sample.selected.tolist() == [True, False, False]


def test_bin_total_counts_eligible_population():
    galaxies = make_population(n=200)
    sample = select_galaxies(galaxy_data=galaxies, params=make_params(), redshift=0.0)
    expected = np.array([(sample.bin_idx == b).sum() for b in range(len(sample.bin_labels))])
    assert np.array_equal(sample.bin_total, expected)


def test_binned_draw_counts_and_short_bins():
    galaxies = make_population(n=200)
    sample = select_galaxies(galaxy_data=galaxies, params=make_params(galaxies_per_bin=5), redshift=0.0)
    for b in range(len(sample.bin_labels)):
        n_selected = (sample.selected & (sample.bin_idx == b)).sum()
        assert n_selected == min(5, sample.bin_total[b])  # short bins take everyone


def test_selection_is_deterministic_and_seed_dependent():
    galaxies = make_population(n=500)
    first = select_galaxies(galaxy_data=galaxies, params=make_params(seed=1), redshift=0.0)
    again = select_galaxies(galaxy_data=galaxies, params=make_params(seed=1), redshift=0.0)
    other = select_galaxies(galaxy_data=galaxies, params=make_params(seed=2), redshift=0.0)
    assert np.array_equal(first.selected, again.selected)
    assert not np.array_equal(first.selected, other.selected)


def test_random_mode_draws_requested_number():
    galaxies = make_population(n=100)
    sample = select_galaxies(galaxy_data=galaxies, params=make_params(mode="RANDOM"), redshift=0.0)
    assert sample.n_selected == 3


def test_explicit_mode_ignores_criteria():
    galaxies = make_galaxies(log_mass_star=[10.2, 12.0], log_ssfr=[-10.0, -10.0])  # galaxy 1 out of mass range
    params = make_params(mode="EXPLICIT", explicit_indices=np.array([1]))
    sample = select_galaxies(galaxy_data=galaxies, params=params, redshift=0.0)
    assert sample.indices.tolist() == [1]


@pytest.mark.parametrize("indices", [[5], [-1], [0, 0]])
def test_explicit_mode_rejects_bad_indices(indices):
    galaxies = make_population(n=3)
    params = make_params(mode="EXPLICIT", explicit_indices=np.array(indices))
    with pytest.raises((IndexError, ValueError)):
        select_galaxies(galaxy_data=galaxies, params=params, redshift=0.0)


def test_outputs_are_full_catalogue_length():
    galaxies = make_population(n=37)
    sample = select_galaxies(galaxy_data=galaxies, params=make_params(), redshift=0.0)
    for array in (sample.selected, sample.eligible, sample.ssfr_class, sample.bin_idx):
        assert len(array) == 37


def test_find_central_galaxies():
    field_halo_index = np.array([0, 0, 1, -1, 1, 2])
    mass_baryon = np.array([5.0, 9.0, 3.0, 100.0, 3.0, 1.0])
    # halo 0 -> galaxy 1 (heavier); halo 1 tie -> galaxy 2 (lower index); orphan never central
    expected = [False, True, True, False, False, True]
    assert _find_central_galaxies(field_halo_index=field_halo_index, mass_baryon=mass_baryon).tolist() == expected


# pipeline stage + standalone selection


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


def test_stage_writes_absorption_columns(absorption_run):
    catalogue_path, _ = absorption_run
    with h5py.File(catalogue_path, "r") as f:
        galaxy_data = f["galaxy_data"]
        absorption = galaxy_data["properties/absorption"]
        n_galaxies = len(galaxy_data["GalID"])

        for name in ("absorption_ssfr_class", "absorption_eligible", "absorption_bin_idx"):
            assert absorption[name].shape == (n_galaxies,)

        assert set(np.unique(absorption["absorption_ssfr_class"][:])) <= {-1, 0, 1, 2}
        assert set(np.unique(absorption["absorption_eligible"][:])) <= {0, 1}
        assert np.all(absorption["absorption_bin_idx"][:] >= -1)


def test_make_spectra_writes_selection(absorption_run):
    catalogue_path, config = absorption_run
    config = replace(config, absorption_selection_only=True)

    selection_path = make_spectra(config=config)  # catalogue path derived from the config
    assert selection_path.exists()

    with h5py.File(catalogue_path, "r") as cat, h5py.File(selection_path, "r") as sel:
        n_galaxies = len(cat["galaxy_data/GalID"])
        for name in ("selected", "eligible", "ssfr_class", "bin_idx"):
            assert sel[name].shape == (n_galaxies,)
        assert len(sel["bin_labels"]) == len(sel["bin_total"])
        assert sel.attrs["mode"] == config.galaxy_selection


def test_make_spectra_is_reproducible(absorption_run):
    _, config = absorption_run
    config = replace(config, absorption_selection_only=True)

    first_path = make_spectra(config=config)
    with h5py.File(first_path, "r") as f:
        first = f["selected"][:]

    second_path = make_spectra(config=config)  # also checks the 'w' overwrite works
    with h5py.File(second_path, "r") as f:
        assert np.array_equal(first, f["selected"][:])
