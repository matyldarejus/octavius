"""

Galaxy sampler for the absorption line analysis module. Galaxies are chosen based on a number of selection schemes
specified in the config. This takes in plan containers so runs identically in-pipeline and on a finished catalogue.

Every galaxy is classified by sSFR and assigned a galaxy bin; the sampling mode then determines which galaxies
are selected:

- ALL:      every eligible galaxy
- RANDOM:   n_galaxies_random drawn uniformly from the eligible galaxies.
- BINNED:   galaxies_per_bin drawn from each galaxy bin in stellar mass - sSFR space.
- EXPLICIT: catalogue galaxy indices provided, here criteria are NOT APPLIED.

The sampler supports two modes of sSFR classification:

- SSFR_CUT:     calculate star-forming sSFR threshold using log sSFR = ssfr_intercept + ssfr_redshift_slope * z;
                green valley within green_valley_dex below it.
                As in Appleby et al. 2023 and most previous Simba work.

- MS_OFFSET:    sSFR threshold calculated using log sSFR = (ms_slope - 1) * log_mass_star + ms_intercept.
                As in Belfiore et al. 2018.

- NONE:         no sSFR classification; galaxy bins are stellar mass ONLY.


All schemes with exception of 'Explicit' sampling select galaxy based on a number of criteria.

"""

# default packages

# other packages
import numpy as np

from ..log import get_logger

# internal imports
from .absorption_helpers import SSFR_CLASS_IDX, SSFR_CLASS_NAMES, GalaxyData, GalaxySample, SamplingParams

logger = get_logger()

LOG_SSFR_FLOOR: float = -30.0  # guard for SFR below zero


# compute necessary properties


def compute_log_mass_star(*, mass_star: np.ndarray) -> np.ndarray:
    """

    Computes log10 stellar mass.

    Returns:
    - log_mass_star: (n_gal,) log10 Msun

    """
    log_mass_star = np.full(len(mass_star), -np.inf, dtype=np.float64)
    valid = mass_star > 0.0
    log_mass_star[valid] = np.log10(mass_star[valid])
    return log_mass_star


def compute_log_ssfr(*, sfr: np.ndarray, mass_star: np.ndarray) -> np.ndarray:
    """

    Computes log10 specific SFR. Galaxies with SFR below zero or zero mass get assigned
    LOG_SSFR_ZERO_SFR.

    Returns:
    - log_ssfr: (n_gal,) log10 yr^-1

    """
    log_ssfr = np.full(len(sfr), LOG_SSFR_FLOOR, dtype=np.float64)
    valid = (sfr > 0.0) & (mass_star > 0.0)
    log_ssfr[valid] = np.log10(sfr[valid] / mass_star[valid])
    return log_ssfr


def compute_ssfr_thresholds(
    *, log_mass_star: np.ndarray, redshift: float, params: SamplingParams
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """

    Computes the star-forming threshold and the lower bound of the green valley class.
    See above for explanation of the two schemes.

    Returns:
    - sf_threshold: log10 yr^-1, lower bound of the star-forming class
    - gv_floor: log10 yr^-1, lower bound of the green-valley class

    """
    if params.ssfr_classification == "SSFR_CUT":
        sf_threshold = params.ssfr_intercept + params.ssfr_redshift_slope * redshift
        gv_floor = sf_threshold - params.green_valley_width

    elif params.ssfr_classification == "MS_OFFSET":
        log_ssfr_ms = (params.ms_slope - 1.0) * log_mass_star + params.ms_intercept
        sf_threshold = log_ssfr_ms - params.gv_n_sigma * params.ms_scatter
        gv_floor = log_ssfr_ms - params.gv_n_sigma * params.ms_scatter

    else:
        raise ValueError(f"Unknown ssfr_classification '{params.ssfr_classification}' for SF threshold computation.")

    return sf_threshold, gv_floor


def classify_ssfr(*, galaxy_data: GalaxyData, redshift: float, params: SamplingParams) -> np.ndarray:
    """

    Classifies galaxies as star-forming, green-valley, or quenched. Galaxies without a finite stellar mass,
    SFR, or SFR below the green-valley floor are set as UNCLASSIFIED.

    quenched_definition:
    - SFR_ZERO:           quenched means exactly zero SFR (default).
    - BELOW_GREEN_VALLEY: quenched means zero SFR or below the green-valley floor.

    """
    # all galaxies are given "UNCLASSIFIED" class by default
    ssfr_class = np.full(galaxy_data.n_galaxies, SSFR_CLASS_IDX["UNCLASSIFIED"], dtype=np.int64)

    if params.ssfr_classification == "NONE":
        return ssfr_class

    sf_threshold, gv_floor = compute_ssfr_thresholds(
        log_mass_star=galaxy_data.log_mass_star, redshift=redshift, params=params
    )

    log_ssfr = galaxy_data.log_ssfr

    # set conditions
    has_mass = np.isfinite(galaxy_data.log_mass_star)
    finite_sfr = np.isfinite(galaxy_data.sfr)  # !!!! check if sfr is NaN for gasless galaxies
    is_zero_sfr = has_mass & finite_sfr & (galaxy_data.sfr <= 0.0)
    has_sfr = has_mass & finite_sfr & (galaxy_data.sfr > 0.0)

    is_sf = has_sfr & (log_ssfr >= sf_threshold)
    is_gv = has_sfr & (log_ssfr < sf_threshold) & (log_ssfr >= gv_floor)

    if params.quenched_definition == "SFR_ZERO":
        is_q = is_zero_sfr
    elif params.quenched_definition == "BELOW_GREEN_VALLEY":
        is_q = is_zero_sfr | (has_sfr & (log_ssfr < gv_floor))
    else:
        raise ValueError(f"Unknown quenched_definition '{params.quenched_definition}'.")

    ssfr_class[is_sf] = SSFR_CLASS_IDX["STAR_FORMING"]
    ssfr_class[is_gv] = SSFR_CLASS_IDX["GREEN_VALLEY"]
    ssfr_class[is_q] = SSFR_CLASS_IDX["QUENCHED"]
    return ssfr_class


# handle binning


def apply_criteria(*, galaxy_data: GalaxyData, params: SamplingParams) -> np.ndarray:
    """
    Applies hard selection criteria, excluding galaxies, rather than binning them.

    In short, this pre-selects the galaxies to be sampled from.

    Returns:
    - eligible: (n_gal,) boolean mask of galaxies entering the sampling pool.
    """
    edges = params.mass_bin_edges
    log_mass_star = galaxy_data.log_mass_star

    eligible = np.isfinite(log_mass_star) & (log_mass_star >= edges[0]) & (log_mass_star < edges[-1])
    eligible &= np.isfinite(galaxy_data.r200c) & (galaxy_data.r200c > 0.0)

    if params.centrals_only:
        eligible &= galaxy_data.is_central

    return eligible


def assign_bins(
    *,
    log_mass_star: np.ndarray,
    ssfr_class: np.ndarray,
    eligible: np.ndarray,
    mass_bin_edges: np.ndarray,
    bin_by_ssfr: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """

    Assigns each eligible galaxy to a mass (and sSFR class) bin.

    Bins are assigned ids with the convention being:
    bin_id = mass_idx * n_classes + ssfr_class, when binning by sSFR class, else bin_id = mass idx.

    """
    n_mass_bins = len(mass_bin_edges) - 1
    mass_idx = np.searchsorted(mass_bin_edges, log_mass_star, side="right") - 1
    mass_idx[log_mass_star == mass_bin_edges[-1]] = n_mass_bins - 1  # close the last bin
    valid = eligible & (mass_idx >= 0) & (mass_idx < n_mass_bins)

    mass_labels = [f"logM_{mass_bin_edges[i]:.3f}-{mass_bin_edges[i + 1]:.3f}" for i in range(n_mass_bins)]

    if bin_by_ssfr:
        n_classes = len(SSFR_CLASS_NAMES)
        valid &= ssfr_class >= 0  # discard unclassified galaxies
        bin_id = mass_idx * n_classes + ssfr_class
        bin_labels = tuple(f"{m}_{c}" for m in mass_labels for c in SSFR_CLASS_NAMES)
    else:
        bin_id = mass_idx
        bin_labels = tuple(mass_labels)

    bin_idx = np.full(len(log_mass_star), -1, dtype=np.int16)
    bin_idx[valid] = bin_id[valid]
    return bin_idx, bin_labels


def count_bins(*, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """

    Counts the eligible population of each galaxy bin. This is stored for reweighing.

    """
    return np.bincount(bin_idx[bin_idx >= 0], minlength=n_bins).astype(np.int64)


def classify_and_bin(
    *, galaxy_data: GalaxyData, params: SamplingParams, redshift: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """

    Classifies every galaxy by sSSFR, applies the selection criteria, and assigns bins, all-in-one.

    This is done purely per-galaxy, so should be safe on rank-local galaxies.

    Returns:
    - ssfr_class: (n_gal,) SSFR_CLASS_IDX values
    - eligible: (n_gal,) array of bools based on criteria met or not
    - bin_idx: (n_gal,) bin indices, defaults to -1 for no assignment
    - bin_labels: (n_bin,) labels

    """
    ssfr_class = classify_ssfr(galaxy_data=galaxy_data, redshift=redshift, params=params)
    eligible = apply_criteria(galaxy_data=galaxy_data, params=params)

    bin_idx, bin_labels = assign_bins(
        log_mass_star=galaxy_data.log_mass_star,
        ssfr_class=ssfr_class,
        eligible=eligible,
        mass_bin_edges=params.mass_bin_edges,
        bin_by_ssfr=params.ssfr_classification != "NONE",
    )
    return ssfr_class, eligible, bin_idx, bin_labels


def draw_without_replacement(
    *, candidates: np.ndarray, n_draw: int, rng: np.random.Generator, label: str
) -> np.ndarray:
    """

    Draw without replacement n_draw catalogue indices from candidates.

    If there are too few eligible galaxies, grab all of them with a warning.

    """
    if len(candidates) <= n_draw:
        if len(candidates) < n_draw:
            logger.warning(f"{label}: {len(candidates)} eligible galaxies for {n_draw} requested; taking all.")
        return candidates

    return np.sort(rng.choice(candidates, size=n_draw, replace=False))


def draw_binned(
    *, bin_idx: np.ndarray, bin_labels: tuple[str, ...], n_per_bin: int, rng: np.random.Generator
) -> np.ndarray:
    """

    Draws n_per_bin galaxies from each galaxy bin, in bin order, from a single generator.

    """
    selected = np.zeros(len(bin_idx), dtype=bool)

    for b, label in enumerate(bin_labels):
        members = np.flatnonzero(bin_idx == b)  # done in catalogue order for reproducibility
        selected[draw_without_replacement(candidates=members, n_draw=n_per_bin, rng=rng, label=label)] = True

    return selected


# selection/sampling functions


def _select_explicit(*, explicit_indices: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """

    Selects explicit catalogue indices without applying any criteria. Useful if analysis is to be ran on a
    specific set of galaxies. Galaxies which do not meet the selection criteria are kept anyway with a warning.

    """
    indices = np.asarray(explicit_indices, dtype=np.int64)
    n_galaxies = len(eligible)

    if np.any((indices < 0) | (indices >= n_galaxies)):  # check if enough indices provided
        raise IndexError(f"Explicit galaxy indices must lie in [0, {n_galaxies}]).")
    if len(np.unique(indices)) != len(indices):  # check for duplicates
        raise IndexError("Explicit galaxy indices contain duplicates.")

    selected = np.zeros(n_galaxies, dtype=bool)
    selected[indices] = True

    n_ineligible = int((selected & ~eligible).sum())
    if n_ineligible > 0:
        logger.warning(f"{n_ineligible} explicit galaxies selected do not pass the selection criteria (kept anyway).")

    return selected


def select_galaxies(*, galaxy_data: GalaxyData, params: SamplingParams, redshift: float) -> GalaxySample:
    """

    Classifies, bins, and samples galaxies based on the sampling parameters.

    Parameters
    -------------
    galaxy_data: GalaxyData
        Per-galaxy quantities in catalogue index order.
    params: SamplingParams
        Sampling mode, criteria, galaxy bins and sSFR classification parameters.
    redshift: float
        Snapshot redshift (for the redshift-dependent sSFR threshold i.e. SSFR_CUT mode).

    Returns
    -------
    sample: GalaxySample
        Full-length per-galaxy selection, sSFR class and galaxy-bin index, plus per-bin labels and
        eligible totals.

    """
    # 1) classify galaxies, apply criteria, and bin

    ssfr_class, eligible, bin_idx, bin_labels = classify_and_bin(
        galaxy_data=galaxy_data, params=params, redshift=redshift
    )

    bin_total = count_bins(bin_idx=bin_idx, n_bins=len(bin_labels))
    rng = np.random.default_rng(params.seed)

    # 2) sample
    if params.mode == "ALL":
        selected = eligible.copy()

    elif params.mode == "RANDOM":
        selected = np.zeros(galaxy_data.n_galaxies, dtype=bool)
        chosen = draw_without_replacement(
            candidates=np.flatnonzero(eligible),
            n_draw=params.n_galaxies_random,
            rng=rng,
            label="RANDOM",
        )
        selected[chosen] = True

    elif params.mode == "BINNED":
        selected = draw_binned(bin_idx=bin_idx, bin_labels=bin_labels, n_per_bin=params.galaxies_per_bin, rng=rng)

    elif params.mode == "EXPLICIT":
        selected = _select_explicit(explicit_indices=params.explicit_indices, eligible=eligible)

    else:
        raise ValueError(f"Unknown galaxy sampling mode '{params.mode}'.")

    logger.info(f"Galaxy sampling ({params.mode}): {int(selected.sum())} / {galaxy_data.n_galaxies} selected.")

    return GalaxySample(
        selected=selected,
        ssfr_class=ssfr_class,
        bin_idx=bin_idx,
        bin_labels=bin_labels,
        bin_total=bin_total,
        mode=params.mode,
    )
