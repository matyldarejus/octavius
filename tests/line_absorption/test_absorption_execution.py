"""

Tests the line absorption stage executor: central galaxies, config parsing and the pipeline stage output.

"""

# default libraries
from pathlib import Path

# other packages
import h5py
import numpy as np

# testing
import pytest

# internal imports
from octavius.data_management.conventions import OctaviusConfig
from octavius.line_absorption.absorption_execution import _find_central_galaxies, _prepare_sampling_params
from octavius.run_octavius import analyse_snapshot
from octavius.utils.generate_snapshots import generate_simba_snapshot

CONFIG_PATH = Path(__file__).parents[2] / "octavius" / "config.yaml"
STAGES = {
    "find_galaxies": True,
    "properties_core": True,
    "properties_ptype_specific": True,
    "properties_local_environment": False,
    "photometry": False,
    "line_absorption": True,
}


def test_find_central_galaxies():
    field_halo_index = np.array([0, 0, 1, -1, 1, 2])
    mass_baryon = np.array([5.0, 9.0, 3.0, 100.0, 3.0, 1.0])
    # halo 0 -> galaxy 1 (heavier); halo 1 tie -> galaxy 2 (lower index); orphan never central
    expected = [False, True, True, False, False, True]
    assert _find_central_galaxies(field_halo_index=field_halo_index, mass_baryon=mass_baryon).tolist() == expected


@pytest.mark.parametrize("edges", [[10.0], [10.0, 10.0, 11.0], [11.0, 10.0]])
def test_prepare_sampling_params_rejects_bad_edges(edges):
    config = OctaviusConfig.from_yaml(config_path=CONFIG_PATH, photometry_table_path=None, mass_bin_edges=edges)
    with pytest.raises(ValueError):
        _prepare_sampling_params(config=config)


@pytest.fixture(scope="module")
def absorption_catalogue(tmp_path_factory: pytest.TempPathFactory):
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

    with h5py.File(analyse_snapshot(config=config), "r") as f:
        yield f


def test_stage_writes_absorption_columns(absorption_catalogue):
    galaxy_data = absorption_catalogue["galaxy_data"]
    absorption = galaxy_data["properties/absorption"]
    n_galaxies = len(galaxy_data["GalID"])

    for name in ("absorption_ssfr_class", "absorption_eligible", "absorption_bin_idx"):
        assert absorption[name].shape == (n_galaxies,)

    assert set(np.unique(absorption["absorption_ssfr_class"][:])) <= {-1, 0, 1, 2}
    assert set(np.unique(absorption["absorption_eligible"][:])) <= {0, 1}
    assert np.all(absorption["absorption_bin_idx"][:] >= -1)
