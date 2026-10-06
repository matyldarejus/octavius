"""

Line absorption tools to run on a merged catalogue.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import OctaviusConfig
    from ..utils.loader import OctaviusCatalogue

# default packages
from pathlib import Path

# other packages
import numpy as np

from ..data_management import output_catalogue_path
from ..log import get_logger
from ..utils import load_catalogue

# internal imports
from .absorption_execution import prepare_sampling_params
from .absorption_helpers import GalaxyData, GalaxySample
from .absorption_output import selection_file_path, write_selection
from .galaxy_sampling import build_galaxy_data, select_galaxies

logger = get_logger()

# could change so defining these here
STORED_COLUMNS: tuple[str, ...] = ("absorption_ssfr_class", "absorption_eligible", "absorption_bin_idx")


def make_spectra(*, config: OctaviusConfig, catalogue_path: Path | None = None) -> Path:
    """

    Gets a galaxy sample, prepares sightlines, and builds mock absorption spectra
    around a galaxy sample drawn from a catalogue.

    Roadmap:

    [x] Galaxy sampling
    [ ] Sightline prep
    [ ] Spectra generation
    [ ] Spectra fitting

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
    params = prepare_sampling_params(config=config)

    if catalogue_path is None:
        catalogue_path = output_catalogue_path(snapshot_path=config.snapshot_path, output_dir=config.output_dir)
    if not catalogue_path.exists():
        raise FileNotFoundError(f"No Octavius catalogue at {catalogue_path}; run analyse_snapshot first.")

    with load_catalogue(catalogue_path=catalogue_path) as catalogue:
        galaxy_data = _prepare_catalogue_galaxy_data(catalogue=catalogue)
        redshift = float(catalogue.sim_info("redshift"))
        stored = _read_stored_classification(catalogue=catalogue)

    sample = select_galaxies(galaxy_data=galaxy_data, params=params, redshift=redshift)

    if stored is not None:
        _warn_on_mismatch(sample=sample, stored=stored)

    selection_path = selection_file_path(snapshot_path=config.snapshot_path, output_dir=config.output_dir)
    if config.absorption_write_selection:
        write_selection(
            path=selection_path, sample=sample, params=params, redshift=redshift, catalogue_path=catalogue_path
        )

    if config.absorption_selection_only:
        logger.info("absorption_selection_only is set; stopping after galaxy selection.")
        return selection_path

    logger.warning("Sightlines and spectra are not implemented yet; stopping after galaxy selection.")  # TODO: C6
    return selection_path


def _prepare_catalogue_galaxy_data(*, catalogue: OctaviusCatalogue) -> GalaxyData:
    """

    Builds GalaxyData from a merged catalogue.

    Returns:
    - galaxy_data: GalaxyData

    """
    galaxies, haloes = catalogue.galaxies, catalogue.haloes
    n_galaxies = len(galaxies)

    mass_star = galaxies.get_dataset("mass_star")  # (n_gal,) Msun
    sfr = galaxies.get_dataset("sfr")  # (n_gal,) Msun/yr
    field_halo_index = galaxies.get_membership("field_halo_index")  # (n_gal,) index into halo_data, -1 if none
    central_galaxy_index = haloes.get_membership(
        "central_galaxy_index"
    )  # (n_haloes,) index into galaxy_data, -1 if none
    halo_r200c = haloes.get_dataset("r200c")  # (n_haloes,) kpc a

    has_halo = field_halo_index >= 0
    host = field_halo_index[has_halo]

    r200c = np.full(n_galaxies, np.nan)  # (n_gal,) kpc a
    r200c[has_halo] = halo_r200c[host]

    is_central = np.zeros(n_galaxies, dtype=bool)
    is_central[has_halo] = central_galaxy_index[host] == np.flatnonzero(has_halo)

    return build_galaxy_data(mass_star=mass_star, sfr=sfr, r200c=r200c, is_central=is_central)


def _read_stored_classification(*, catalogue: OctaviusCatalogue) -> dict[str, np.ndarray] | None:
    """

    Reads the absorption columns prepared in-pipeline.

    Returns:
    - stored: dictionary of the three columns (ssfr_class, eligible, bin_idx), or None if absent

    """
    keys = set(catalogue.galaxies.keys())
    if not all(name in keys for name in STORED_COLUMNS):
        return None
    return {name: catalogue.galaxies.get_dataset(name) for name in STORED_COLUMNS}


def _warn_on_mismatch(*, sample: GalaxySample, stored: dict[str, np.ndarray]) -> None:
    """

    Warns if recomputed classification is different than in the catalogue e.g. if config was changed since the pipeline was ran.

    """
    recomputed = {
        "absorption_ssfr_class": sample.ssfr_class,
        "absorption_eligible": sample.eligible.astype(np.int64),
        "absorption_bin_idx": sample.bin_idx,
    }
    for name, values in recomputed.items():
        n_differ = int(np.count_nonzero(stored[name] != values))
        if n_differ > 0:
            logger.warning(f"{name}: {n_differ} galaxies differ from the catalogue; using recomputed values.")
