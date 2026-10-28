"""

Processing of synthesised optical depths into mock spectra.

The instrument blurs the transmitted flux, and not the tau itself, so all computations are done on F.


"""

# default libraries
import math

# other packages
import numpy as np
from astropy import units as u
from astropy.constants import codata2022 as codata
from scipy.ndimage import convolve1d
from scipy.special import erf

# internal imports
from ..log import get_logger

logger = get_logger()

# precalculate / set constants
C_KMS = codata.c.to(u.km / u.s).value
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
N_LSF_SIGMA = 5.0  # no. gaussian standard deviations


def gaussian_lsf(*, resolving_power: float, dv: float) -> np.ndarray:
    """

    Gaussian LFS with FWHM = c / R; fraction of the kernel falling in each pixel of width dv,
    centred on the middle pixel and extended out to +/- N_LSF_SIGMA.

    """
    fwhm = C_KMS / resolving_power
    if fwhm < 2.0 * dv:
        logger.warning(f"LSF FWHM {fwhm:.2f} km.s is under two pixels of {dv:.2f} km/s; the LSF is undersampled.")
    sigma = fwhm / FWHM_PER_SIGMA
    n_half = math.ceil(N_LSF_SIGMA * sigma / dv)
    edges = (np.arange(-n_half, n_half + 2) - 0.5) * dv
    kernel = np.diff(0.5 * erf(edges / (math.sqrt(2.0) * sigma)))
    return kernel / kernel.sum()


def apply_lsf(*, flux: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """

    Convolves each spectrum with the LSF kernel, wrapping periodically in velocity.

    Currently done with scipy.convolve1d, cost per spectrum = N_pix * kernel_length, might be deprecated in
    future.

    """
    if len(kernel) > flux.shape[-1]:
        raise ValueError(f"LSF kernel ({len(kernel)} pixels) is wider than the spectrum ({flux.shape[-1]} pixels).")
    return convolve1d(flux, kernel, axis=-1, mode="wrap")
