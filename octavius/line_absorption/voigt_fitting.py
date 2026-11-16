"""

Voigt profile fitting of absorption spectra.

Based on the pygad vpfit module (Davé & Lilje 2026).

Original VPFIT routine is described by Davé & Hernquist et al. (1997) (doi: 10.1086/306722).

"""

# default libraries
import math


# other packages
import numpy as np
from scipy.ndimage import convolve1d


# internal imports
from ..log import get_logger
from .absorption_computations import deposit_line, profile_half_width
from .absorption_helpers import LineConstants
from .spectrum_processing import apply_lsf

logger = get_logger()

DETECTION_WIDTHS = range(2, 11)  # Gaussian widths in pixels
KERNEL_TRUNCATION = 6.0  # filter half-width in units of its width
REGION_MERGE_GAP = 5  # regions closer than this number of pixels are merged
REGION_PAD = 2  # number of pixels to be added on to each side of a region


def detection_ratio(*, flux: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """

    Calculates detection ratio of absorption at each pixel: the absorbed flux 1 - F smoothed by a Gaussian over
    its propagated noise, maximised over the widths in DETECTION_WIDTHS. Depths within the noise do not count.

    """
    absorbed = np.where(1.0 - flux >= noise, 1.0 - flux, 0.0)
    ratio = np.zeros(len(flux))
    for width in DETECTION_WIDTHS:
        offsets = np.arange(-math.ceil(KERNEL_TRUNCATION * width), math.ceil(KERNEL_TRUNCATION * width) + 1)
        kernel = np.exp(-0.5 * (offsets / width) ** 2)
        signal = convolve1d(absorbed, kernel, mode="constant", cval=0.0)
        variance = convolve1d(noise**2, kernel**2, mode="constant", cval=0.0)
        ratio = np.maximum(ratio, signal / np.sqrt(variance))
    return ratio


def merge_regions(*, starts: np.ndarray, ends: np.ndarray, n_pixels: int) -> np.ndarray:
    """

    Merges regions closer than REGION_MERGE_GAP and pads each by REGION_PAD, within the spectrum and
    overlapping its neighbours.

    """
    merged: list[list[int]] = []
    for start, end in zip(starts, ends):  # get region
        if merged and start - merged[-1][1] < REGION_MERGE_GAP:  #  check gap between end of last 'merged' region
            merged[-1][1] = int(end)  # move end of the merged region
        else:
            merged.append([int(start), int(end)])  # otherwise add region as new entry, inc. first region

    regions = np.array(merged, dtype=np.int64).reshape(-1, 2)
    regions[:, 0] = np.maximum(regions[:, 0] - REGION_PAD, 0)  # prevent padding if near start
    regions[:, 1] = np.minimum(regions[:, 1] + REGION_PAD, n_pixels)  # prevent going over end edge
    regions[1:, 0] = np.maximum(regions[1:, 0], regions[:-1, 1])  # make sure neighbours never share pixels
    return regions


def find_absorption_regions(*, flux: np.ndarray, noise: np.ndarray, n_sigma: float) -> np.ndarray:
    """

    Finds the regions in a spectrum where absorption is particularly strong, as an array of half-open pixel ranges.

    A region opens at a pixel with flux below the continuum and a positive detection ratio, and then stays open
    until the flux rises above the contiuum - it is kept if its ratios add in quadrature to more than n_sigma.

    """
    n_pixels = len(flux)
    ratio = detection_ratio(flux=flux, noise=noise)

    # look for flux "walls" where it's above 1.0
    above = flux > 1.0
    stretch = np.cumsum(above)  # get the number of wals
    stretch_ends = np.append(np.flatnonzero(above), n_pixels)

    # open each stretch at first suitable pixel
    opening = np.flatnonzero((ratio > 0.0) & (flux < 1.0))
    stretches, first = np.unique(stretch[opening], return_index=True)
    starts, ends = opening[first], stretch_ends[stretches]

    # keep the significant ones
    cumulative = np.concatenate(([0.0], np.cumsum(ratio**2)))
    significant = np.sqrt(cumulative[ends] - cumulative[starts]) > n_sigma

    return merge_regions(starts=starts[significant], ends=ends[significant], n_pixels=n_pixels)


def component_tau(
    *, log_column: float, b: float, v: float, line: LineConstants, n_pixels: int, dv: float
) -> np.ndarray:
    """

    Optical depth of one component (log10 N in cm^-2, Doppler parameter b and centre v in km/s) on a periodic grid
    of n_pixels pixels of width dv, deposited the same way as when synthesis (see notes in
    absorption_computations.deposit_line).

    """
    column = 10.0**log_column
    tau = np.zeros(n_pixels)
    half_width = profile_half_width(column, b, line.xsec, line.gamma)
    deposit_line(tau, np.zeros((0, n_pixels)), np.zeros(0), column, v, b, line.gamma, dv, half_width)
    return tau * (line.xsec / dv)


def model_flux(*, components: np.ndarray, line: LineConstants, n_pixels: int, dv: float, kernel: np.ndarray):
    """

    Model flux of the components (rows of log10 N, b, v) as observed: exp(-tau) of their summed optical depth,
    convolved with the LSF kernel.

    """
    tau = np.zeros(n_pixels)
    for log_column, b, v in components:
        tau += component_tau(log_column=log_column, b=b, v=v, line=line, n_pixels=n_pixels, dv=dv)
    return apply_lsf(flux=np.exp(-tau)[None, :], kernel=kernel)[0]  # works on rows, so need to fix with indexing
