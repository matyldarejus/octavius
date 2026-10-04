"""

Tests that the line_absorption config section parses and validates.

"""

# default libraries
from pathlib import Path

# testing
import pytest

# internal imports
from octavius.data_management.conventions import OctaviusConfig

CONFIG_PATH = Path(__file__).parents[2] / "octavius" / "config.yaml"


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
