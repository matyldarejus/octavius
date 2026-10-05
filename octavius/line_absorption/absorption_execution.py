"""

Functions to handle executing the line absorption stage.

Currently under construction!!

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import GroupStore, OctaviusConfig, SimulationData

# other packages
import numpy as np

from ..log import get_logger

# internal imports
from .absorption_helpers import GalaxyData, SamplingParams
from .galaxy_sampling import classify_and_bin, compute_log_mass_star, compute_log_ssfr

logger = get_logger()


def run_absorption(simulation_data: SimulationData, config: OctaviusConfig) -> None:
    """
    Executor for absorption line analysis.
    """
    if "galaxies" not in simulation_data.groups:
        logger.info("Skipping absorption line analysis: no galaxies found.")
        return
    logger.info("Preparing absorption line analysis.")

    # read in galaxies and haloes
    galaxies = simulation_data.groups["galaxies"]
    haloes = simulation_data.groups.get("haloes")

    params = _prepare_sampling_params(config=config)
    galaxy_data = _prepare_galaxy_data(galaxies=galaxies, haloes=haloes)

    ssfr_class, eligible, bin_idx, bin_labels = classify_and_bin(
        galaxy_data=galaxy_data, params=params, redshift=simulation_data.simulation.redshift
    )

    results: dict[str, np.ndarray] = {
        "absorption_ssfr_class": ssfr_class,
        "absorption_eligible": eligible,
        "absorption_bin_idx": bin_idx,
    }

    galaxies.write_batch(results=results)

    logger.info(
        f"Line absorption: {int(eligible.sum())} / {galaxy_data.n_galaxies} galaxies eligible "
        f"across {len(bin_labels)} galaxy bins."
    )


def _prepare_sampling_params(*, config: OctaviusConfig) -> SamplingParams:
    """

    Parses the config fields into SamplingParams. Validate mass-bin edges here.

    """
    edges = np.asarray(config.mass_bin_edges, dtype=np.float64)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("mass_bin_edges must contain at least two strictly increasing values.")

    return SamplingParams(
        mode=config.galaxy_selection,
        explicit_indices=np.asarray(config.explicit_galaxy_indices, dtype=np.int64),
        centrals_only=config.centrals_only,
        mass_bin_edges=edges,
        galaxies_per_bin=config.galaxies_per_bin,
        n_galaxies_random=config.n_galaxies_random,
        seed=config.absorption_seed,
        ssfr_classification=config.ssfr_classification,
        quenched_definition=config.quenched_definition,
        ssfr_intercept=config.ssfr_intercept,
        ssfr_redshift_slope=config.ssfr_redshift_slope,
        green_valley_width=config.green_valley_width,
        ms_slope=config.ms_slope,
        ms_intercept=config.ms_intercept,
        ms_scatter=config.ms_scatter,
        sf_n_sigma=config.sf_n_sigma,
        gv_n_sigma=config.gv_n_sigma,
    )


def _prepare_galaxy_data(*, galaxies: GroupStore, haloes: GroupStore | None) -> GalaxyData:
    """

    Builds GalaxyData from the rank-local galaxies and haloes GroupStores.

    field_halo_index lookups are rank-consistent as a field halo and all of its galaxies live on the same rank.

    """
    n_galaxies = galaxies.n_groups
    mass_star = galaxies["mass_star"]
    sfr = galaxies["sfr"]
    field_halo_index = galaxies["field_halo_index"]  # if orphan this will be -1
    has_halo = field_halo_index >= 0

    r200c = np.full(n_galaxies, np.nan)  # kpc a

    # TEMP: check
    if haloes is not None and "radius_200c" in haloes:
        r200c[has_halo] = haloes["radius_200c"][field_halo_index[has_halo]]
    else:
        logger.warning("No 'r200c' present on haloes")

    is_central = _find_central_galaxies(field_halo_index=field_halo_index, mass_baryon=galaxies["mass_baryon"])

    return GalaxyData(
        log_mass_star=compute_log_mass_star(mass_star=mass_star),
        sfr=sfr,
        log_ssfr=compute_log_ssfr(sfr=sfr, mass_star=mass_star),
        r200c=r200c,
        is_central=is_central,
    )


def _find_central_galaxies(*, field_halo_index: np.ndarray, mass_baryon: np.ndarray) -> np.ndarray:
    """

    Flags the central i.e. most massive galaxy in each halo.

    """
    is_central = np.zeros(len(field_halo_index), dtype=bool)  # start with all false
    with_halo = np.flatnonzero(field_halo_index >= 0)  # remove galaxies without haloes
    if with_halo.size == 0:
        return is_central

    halo = field_halo_index[with_halo]  # get galaxy's halo index
    order = np.lexsort((-mass_baryon[with_halo], halo))  # by halo, then mass heaviest first
    sorted_halo = halo[order]  # sort

    first_in_halo = np.ones(len(order), dtype=bool)
    first_in_halo[1:] = sorted_halo[1:] != sorted_halo[:-1]

    # map back to galaxy indices
    is_central[with_halo[order[first_in_halo]]] = True
    return is_central
