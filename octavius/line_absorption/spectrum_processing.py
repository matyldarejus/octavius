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

LEGACY_LSF_DIR = Path(__file__).parent / "data" / "lsf" / "legacy"
LEGACY_LSFS = frozenset(
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

STSCI_LSF_DIR = Path(__file__).parent / "data" / "lsf" / "stsci_cos"
# COS2025 Cycle 33 settings (or the newest published LP where the default is unpublished)
# in order of preference, first setting chosen will be the one which covers wavelength_obs
COS2025_LSFS = (
    "COS_G130M_1291_LP5",
    "COS_G130M_1300_LP5",
    "COS_G130M_1309_LP5",
    "COS_G130M_1318_LP5",
    "COS_G130M_1327_LP5",
    "COS_G130M_1222_LP4",
    "COS_G130M_1096_LP2",
    "COS_G130M_1055_LP2",
    "COS_G160M_1600_LP6",
    "COS_G160M_1577_LP6",
    "COS_G160M_1589_LP6",
    "COS_G160M_1611_LP6",
    "COS_G160M_1623_LP6",
    "COS_G160M_1533_LP6",
    "COS_G140L_1280_LP3",
    "COS_G140L_1105_LP3",
    "COS_G140L_800_LP4",
)

# echelle tables are tabulated in angstrom at a single wavelenght, but the true LSF is fixed in velocity
# hence, when converting to velocity, reference wavelength needs to be used instead of spectrum's own wavelength
ECHELLE_REFERENCE_WAVELENGTH = {"STIS_E140M": 1430.5, "STIS_E140H": 1425.0, "STIS_E230M": 2340.0, "STIS_E230H": 2377.0}


def lsf_kernel(*, lsf: str, dv: float, wavelength_obs: float, resolving_power: float) -> np.ndarray:
    """

    Returns the LSF kernel named in the config, on a velocity grid of pixel width dv; kernel of ones for NONE,
    a Gaussian with resolving power R for GAUSSIAN, or the tabulated LSF of a grating at wavelength_obs in Angstrom.

    """
    if lsf == "NONE":
        return np.ones(1)
    if lsf == "GAUSSIAN":
        return gaussian_lsf(resolving_power=resolving_power, dv=dv)
    if lsf == "COS":
        lsf = cos_setting(wavelength_obs=wavelength_obs)
        logger.info(f"COS setting {lsf} for {wavelength_obs:.1f} A.")
    return tabulated_lsf(grating=lsf, wavelength_obs=wavelength_obs, dv=dv)


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


def load_tabulated_lsf(*, grating: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Reads a tabulated LSF.


    Returns:
        - channels: wavelengths the LSF is tabulated at
        - relwave: wavelength offsets from the line centre
        - table: LSF values, one column per channel

    """
    path = STSCI_LSF_DIR / f"lsf_{grating.lower()}.csv"
    if not path.exists():
        path = LEGACY_LSF_DIR / f"lsf_{grating.lower()}.csv"
    with path.open() as f:
        rows = [line for line in f if not line.startswith("#")]
    channels = np.array(rows[0].split(",")[1:], dtype=float)
    values = np.loadtxt(rows[1:], delimiter=",")
    return channels, values[:, 0], values[:, 1:]


def _covers(channels: np.ndarray, wavelength: float) -> bool:
    """

    True if wavelength lies within the tabulated range and within one channel spacing of a channel
    (i.e. not in a detector gap).

    """
    spacing = np.min(np.diff(channels)) if len(channels) > 1 else np.inf
    inside = channels[0] <= wavelength <= channels[-1]
    return bool(inside and np.min(np.abs(channels - wavelength)) <= spacing)


def cos_setting(*, wavelength_obs: float) -> str:
    """

    Returns first COS2025 setting whose channels cover wavelength_obs, otherwise the setting
    with the nearest channel.

    """
    channels = [load_tabulated_lsf(grating=name)[0] for name in COS2025_LSFS]
    for name, setting_channels in zip(COS2025_LSFS, channels):
        if _covers(setting_channels, wavelength_obs):
            return name
    nearest = np.argmin([np.min(np.abs(c - wavelength_obs)) for c in channels])
    return COS2025_LSFS[nearest]


def tabulated_lsf(*, grating: str, wavelength_obs: float, dv: float) -> np.ndarray:
    """

    Tabulated LSF of a grating at the observed wavelength, as the fraction of the kernel falling in each
    pixel of width dv, centred on the middle pixel, normalised to unit sum.

    Uses the channel nearest wavelength_obs (the closest edge outside the tabulated range). Offsets are
    converted to velocity with v = c * relwave / lambda, where lambda is wavelength_obs for first-order
    gratings and the table's reference wavelength for echelle gratings.


    """
    channels, relwave, table = load_tabulated_lsf(grating=grating)
    if not _covers(channels, wavelength_obs):
        logger.warning(
            f"{grating}: {wavelength_obs:.1f} A is outside the tabulated channels ({channels[0]:.0f}-{channels[-1]:.0f} A) "
            f"or in a detector gap; using the nearest channel."
        )
    profile = table[:, np.argmin(np.abs(channels - wavelength_obs))]

    v_table = C_KMS * relwave / ECHELLE_REFERENCE_WAVELENGTH.get(grating, wavelength_obs)
    cumulative = cumulative_trapezoid(profile, v_table, initial=0.0)
    n_half = math.ceil(v_table[-1] / dv)
    edges = (np.arange(-n_half, n_half + 2) - 0.5) * dv
    kernel = np.diff(np.interp(edges, v_table, cumulative))
    return kernel / kernel.sum()
