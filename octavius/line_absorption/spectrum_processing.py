"""

Processing of synthesised optical depths into mock spectra.

The instrument blurs the transmitted flux, and not the tau itself, so all computations are done on F.


"""

# default libraries
import math
from pathlib import Path

# other packages
import numpy as np
from astropy import units as u
from astropy.constants import codata2022 as codata
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import convolve1d
from scipy.special import erf

# internal imports
from ..log import get_logger

logger = get_logger()

# precalculate / set constants
C_KMS = codata.c.to(u.km / u.s).value
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
N_LSF_SIGMA = 5.0  # no. gaussian standard deviations

PYGAD_LSF_DIR = Path(__file__).parent / "data" / "lsf" / "original"
TABULATED_LSFS = frozenset(
    {
        "COS_G130M",
        "COS_G140L",
        "COS_G160M",
        "STIS_E140H",
        "STIS_E140M",
        "STIS_E230H",
        "STIS_E230M",
        "STIS_G140L",
        "STIS_G140M",
        "STIS_G230L",
        "STIS_G230M",
    }
)

# echelle tables are tabulated in angstrom at a single wavelenght, but the true LSF is fixed in velocity
# hence, when converting to velocity, reference wavelength needs to be used instead of spectrum's own wavelength
ECHELLE_REFERENCE_WAVELENGTH = {"STIS_E140M": 1430.5, "STIS_E140H": 1425.0, "STIS_E230M": 2340.0, "STIS_E230H": 2377.0}


def gaussian_lsf(*, resolving_power: float, dv: float) -> np.ndarray:
    """

    Gaussian LSF with FWHM = c / R; fraction of the kernel falling in each pixel of width dv,
    centred on the middle pixel and extended out to +/- N_LSF_SIGMA.

    """
    fwhm = C_KMS / resolving_power
    if fwhm < 2.0 * dv:
        logger.warning(f"LSF FWHM {fwhm:.2f} km/s is under two pixels of {dv:.2f} km/s; the LSF is undersampled.")
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


def load_tabulated_lsf(*, grating: str, directory: Path = PYGAD_LSF_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Reads a tabulated LSF.


    Returns:
        - channels: wavelengths the LSF is tabulated at
        - relwave: wavelength offsets from the line centre
        - table: LSF values, one column per channel

    """
    with (directory / f"lsf_{grating.lower()}.csv").open() as f:
        rows = [line for line in f if not line.startswith("#")]
    channels = np.array(rows[0].split(",")[1:], dtype=float)
    values = np.loadtxt(rows[1:], delimiter=",")
    return channels, values[:, 0], values[:, 1:]


def tabulated_lsf(*, grating: str, wavelength_obs: float, dv: float) -> np.ndarray:
    """

    Tabulated LSF of a grating at the observed wavelength, as the fraction of the kernel falling in each
    pixel of width dv, centred on the middle pixel, normalised to unit sum.

    Uses the channel nearest wavelength_obs (the closest edge outside the tabulated range). Offsets are
    converted to velocity with v = c * relwave / lambda, where lambda is wavelength_obs for first-order
    gratings and the table's reference wavelength for echelle gratings.


    """
    channels, relwave, table = load_tabulated_lsf(grating=grating)
    if not channels[0] <= wavelength_obs <= channels[-1]:
        logger.warning(
            f"{grating}: {wavelength_obs:.1f} A outside the tabulated channels"
            f"({channels[0]:.0f}-{channels[-1]:.0f}); using the closest edge."
        )
    profile = table[:, np.argmin(np.abs(channels - wavelength_obs))]

    v_table = C_KMS * relwave / ECHELLE_REFERENCE_WAVELENGTH.get(grating, wavelength_obs)
    cumulative = cumulative_trapezoid(profile, v_table, initial=0.0)
    n_half = math.ceil(v_table[-1] / dv)
    edges = (np.arange(-n_half, n_half + 2) - 0.5) * dv
    kernel = np.diff(np.interp(edges, v_table, cumulative))
    return kernel / kernel.sum()
