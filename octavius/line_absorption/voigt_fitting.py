"""

Voigt profile fitting of absorption spectra.

Based on the pygad vpfit module (Davé & Lilje 2026).

Original VPFIT routine is described by Davé & Hernquist et al. (1997) (doi: 10.1086/306722).

"""

# default libraries
import math
from dataclasses import dataclass


# other packages
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import least_squares


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
DERIVATIVE_STEP = 1.0e-3  # central-difference step
BOUND_MARGIN = 1.0e-6  # start parameters are moved by this fraction of their range inside the bounds


@dataclass(frozen=True, slots=True)
class SpectrumToFit:
    """

    Rolled periodic spectrum and all the parameters a fitter may need.

    """

    flux: np.ndarray  # (n_pixels,)
    noise: np.ndarray  # (n_pixels,) 1 sigma
    dv: float  # km/s, pixel width
    line: LineConstants
    kernel: np.ndarray  # LSF kernel applied to the data
    log_column_bounds: tuple[float, float]  # log10 cm^-2
    b_bounds: tuple[float, float]  # km/s

    @property
    def n_pixels(self) -> int:
        return len(self.flux)

    def model(self, components: np.ndarray) -> np.ndarray:
        """

        Model flux of the components on this spectrum's grid.

        """
        return model_flux(components=components, line=self.line, n_pixels=self.n_pixels, dv=self.dv, kernel=self.kernel)


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
        signal = convolve1d(absorbed, kernel, mode="wrap", cval=0.0)
        variance = convolve1d(noise**2, kernel**2, mode="wrap", cval=0.0)
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


def model_flux(
    *, components: np.ndarray, line: LineConstants, n_pixels: int, dv: float, kernel: np.ndarray
) -> np.ndarray:
    """

    Model flux of the components (rows of log10 N, b, v) as observed: exp(-tau) of their summed optical depth,
    convolved with the LSF kernel.

    """
    tau = np.zeros(n_pixels)
    for log_column, b, v in components:
        tau += component_tau(log_column=log_column, b=b, v=v, line=line, n_pixels=n_pixels, dv=dv)
    return apply_lsf(flux=np.exp(-tau)[None, :], kernel=kernel)[0]  # works on rows, so need to fix with indexing


def model_derivatives(*, spectrum: SpectrumToFit, components: np.ndarray) -> np.ndarray:
    """

    Derivatives of the model with respet to every parameter in the order of components.ravel().

    With F = exp(-tau), dF/dp = -F dtau/dp, where

        dtau/dlog 10 N = ln(10) tau; dtau/db and dtau/dv via central differences.

    The LSF convolution is linear, so the derivatives are convolved with the same kernel as the model.

    """
    grid = {"line": spectrum.line, "n_pixels": spectrum.n_pixels, "dv": spectrum.dv}
    taus = np.array([component_tau(log_column=c, b=b, v=v, **grid) for c, b, v in components])
    transmitted = np.exp(-taus.sum(axis=0))

    tau_derivatives = np.empty((3 * len(components), spectrum.n_pixels))
    for i, (log_column, b, v) in enumerate(components):
        h_b, h_v = DERIVATIVE_STEP * b, DERIVATIVE_STEP * spectrum.dv
        tau_derivatives[3 * i] = np.log(10.0) * taus[i]
        tau_derivatives[3 * i + 1] = (
            component_tau(log_column=log_column, b=b + h_b, v=v, **grid)
            - component_tau(log_column=log_column, b=b - h_b, v=v, **grid)
        ) / (2.0 * h_b)
        tau_derivatives[3 * i + 2] = (
            component_tau(log_column=log_column, b=b, v=v + h_v, **grid)
            - component_tau(log_column=log_column, b=b, v=v - h_v, **grid)
        ) / (2.0 * h_v)
    return apply_lsf(flux=-transmitted * tau_derivatives, kernel=spectrum.kernel)


def component_errors(*, jacobian: np.ndarray) -> np.ndarray:
    """

    1 sigma parameter errors from the Jacobian of the noise-weighted residuals via its SVD.
    Parameters with a share in a singular direction are not constrained by the data and get an
    infinite error; the full SVD keeps those directions also when there are more parameters
    than pixels.

    """
    n_parameters = jacobian.shape[1]
    try:
        _, singular, v_t = np.linalg.svd(jacobian, full_matrices=True)
    except np.linalg.LinAlgError:
        return np.full(n_parameters, np.nan)
    singular = np.concatenate((singular, np.zeros(n_parameters - len(singular))))  # missing directions: 0
    if singular[0] == 0.0:
        return np.full(n_parameters, np.inf)

    constrained = singular > np.finfo(float).eps * max(jacobian.shape) * singular[0]
    variance = np.sum((v_t[constrained] / singular[constrained, None]) ** 2, axis=0)
    variance[np.any(np.abs(v_t[~constrained]) > 1e-6, axis=0)] = np.inf
    return np.sqrt(variance)


def fit_components(
    *, spectrum: SpectrumToFit, components: np.ndarray, pixels: slice, v_bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """

    Least-sqaures fit of the components to the spectrum over its pixels, keeping centre i
    within v_bounds[i] (km/s). The model is evaluated over the whole periodic box, so the LSF
    and line wings see the pixels around the region.

    Returns the best-fit components and their 1 sigma errors.

    """
    n_components = len(components)
    if n_components == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    lower = np.column_stack(
        (
            np.full(n_components, spectrum.log_column_bounds[0]),
            np.full(n_components, spectrum.b_bounds[0]),
            v_bounds[:, 0],
        )
    ).ravel()
    upper = np.column_stack(
        (
            np.full(n_components, spectrum.log_column_bounds[1]),
            np.full(n_components, spectrum.b_bounds[1]),
            v_bounds[:, 1],
        )
    ).ravel()
    margin = BOUND_MARGIN * (upper - lower)
    start = np.clip(components.ravel(), lower + margin, upper - margin)

    flux, noise = spectrum.flux[pixels], spectrum.noise[pixels]

    def residuals(x):
        return (flux - spectrum.model(x.reshape(-1, 3))[pixels]) / noise

    def jacobian(x):  # d residual / dp = -dF/dp / sigma
        return -model_derivatives(spectrum=spectrum, components=x.reshape(-1, 3))[:, pixels].T / noise[:, None]

    result = least_squares(residuals, start, jac=jacobian, bounds=(lower, upper), method="trf", x_scale="jac")
    return result.x.reshape(-1, 3), component_errors(jacobian=result.jac).reshape(-1, 3)
