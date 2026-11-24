"""

Voigt profile fitting of absorption spectra.

Based on the pygad vpfit module (Davé & Lilje 2026).

Original VPFIT routine is described by Davé & Hernquist et al. (1997) (doi: 10.1086/306722).

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import OctaviusConfig

# default libraries
import math
import time
from dataclasses import dataclass


# other packages
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.optimize import least_squares


# internal imports
from ..log import get_logger
from .absorption_computations import deposit_line, profile_half_width
from .absorption_helpers import (
    LineConstants,
    SightlineLineLists,
    SightlineParticles,
    Sightlines,
    SightlineSpectra,
    Transition,
)
from .line_profiles import line_constants, load_line_table
from .spectrum_processing import C_KMS, apply_lsf

logger = get_logger()

DETECTION_WIDTHS = range(2, 11)  # Gaussian widths in pixels
KERNEL_TRUNCATION = 6.0  # filter half-width in units of its width
REGION_MERGE_GAP = 5  # regions closer than this number of pixels are merged
REGION_PAD = 2  # number of pixels to be added on to each side of a region
DERIVATIVE_STEP = 1.0e-3  # central-difference step
BOUND_MARGIN = 1.0e-6  # start parameters are moved by this fraction of their range inside the bounds
GUESS_LOG_COLUMNS = 100  # grid points iin log N for a starting guess
GUESS_B_VALUES = 40  # grid points in b for a starting guess
GUESS_FLOOR_SIGMA = 1.5  # a guess may not dip more than this many sigma below the smoothed residual
GUESS_SMOOTHING = 1.0  # pixels, Gaussian width for smoothing the residual
DCHISQ_PER_COMPONENT = 6.0  # a component must lower the total chi2 by this much
MIN_REGION_PIXELS = 4
B_FLOOR_FRACTION = 0.25  # a lower b bound of 0 becomes this fraction
B_FLOOR_TEMPERATURE = 1.0e4  # in Kelvin


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


@dataclass(frozen=True, slots=True)
class LineList:
    """

    Fitted components of one sightline.

    """

    components: np.ndarray  # log10 N in com^-2, b and v in km/s
    errors: np.ndarray  # 1 sigma
    equivalent_width: np.ndarray  # km/s, of eah component alone over its region
    reduced_chisq: np.ndarray  # chi2 per pixel of the full model over each component's region
    region: np.ndarray  # index of each component's region
    regions: np.ndarray  # pixel ranges


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


def saturated_components(
    *, components: np.ndarray, line: LineConstants, n_pixels: int, dv: float, kernel: np.ndarray, tau_threshold: float
) -> np.ndarray:
    """

    Whether each component on its own blacks out the spectrum: its model flux, averaged over pixels and smoothed by
    the LSF, falls below exp(-tau_threshold) somewhere. A component narrower than a pixel or the LSF has a large
    central optical depth, but never a black pixel, so it is not flagged.

    """
    floor = math.exp(-tau_threshold)
    grid = {"line": line, "n_pixels": n_pixels, "dv": dv, "kernel": kernel}
    return np.array(
        [model_flux(components=component[None, :], **grid).min() < floor for component in components], dtype=bool
    )


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


def starting_guess(*, spectrum: SpectrumToFit, residual: np.ndarray, pixels: slice) -> np.ndarray:
    """

    Starting set of parameters for a new component in a region. This is centred on the deepest pixel of the
    smoothed, with the (N, b) on a grid that best matches the residual around the line core without dipping
    more than GUESS_FLOOR_SIGMA below it anywhere in the region. The guess is not convolved with the LSF.

    """
    smoothed = gaussian_filter1d(residual, GUESS_SMOOTHING, mode="wrap")[pixels]  # find the centre
    residual, noise = residual[pixels], spectrum.noise[pixels]
    centre = int(np.argmin(smoothed))
    v = (pixels.start + centre + 0.5) * spectrum.dv
    floor = np.minimum(smoothed, 1.0) - GUESS_FLOOR_SIGMA * noise  # set a floor

    log_columns = np.linspace(*spectrum.log_column_bounds, GUESS_LOG_COLUMNS)  # try every (N, b) on a grid
    b_values = np.geomspace(*spectrum.b_bounds, GUESS_B_VALUES)
    grid = {"line": spectrum.line, "n_pixels": spectrum.n_pixels, "dv": spectrum.dv}

    chisq = np.full((GUESS_B_VALUES, GUESS_LOG_COLUMNS), np.inf)
    for j, b in enumerate(b_values):
        largest = log_columns[-1]  # widest deposit window, then scaled down to every column
        unit_tau = component_tau(log_column=largest, b=b, v=v, **grid)[pixels] / 10.0**largest
        flux = np.exp(-(10.0 ** log_columns[:, None]) * unit_tau)
        depth = 1.0 - flux
        half_depth = depth > 0.5 * depth[:, centre, None]
        half_depth[:, centre] = True  # pygad's window always holds the centre
        core = half_depth.copy()
        core[:, 1:] |= half_depth[:, :-1]  # plus the first pixel past half depth on each side, as in pygad
        core[:, :-1] |= half_depth[:, 1:]
        core_chisq = np.sum(np.where(core, ((residual - flux) / noise) ** 2, 0.0), axis=1) / np.sum(core, axis=1)
        above_floor = np.all(flux >= floor, axis=1)  # throw out everything above the floor
        chisq[j] = np.where(above_floor, core_chisq, np.inf)

    j, i = np.unravel_index(np.argmin(chisq), chisq.shape)  # pick the ones with the lowest score
    return np.array([log_columns[i], b_values[j], v])


def split_component(*, components: np.ndarray, index: int) -> np.ndarray:
    """

    Replaces component index by two with half with its column and 0.7 its b, at v -+ b / 2, appended at the end.

    """
    log_column, b, v = components[index]
    halves = np.array(
        [[log_column - np.log10(2.0), 0.7 * b, v - 0.5 * b], [log_column - np.log10(2.0), 0.7 * b, v + 0.5 * b]]
    )
    return np.vstack((np.delete(components, index, axis=0), halves))


def total_chisq(*, spectrum: SpectrumToFit, components: np.ndarray, pixels: slice) -> float:
    """

    Total chi2 of the components' model over the pixels.

    """
    residuals = (spectrum.flux[pixels] - spectrum.model(components)[pixels]) / spectrum.noise[pixels]
    return float(np.sum(residuals**2))


def fit_in_region(*, spectrum: SpectrumToFit, components: np.ndarray, pixels: slice) -> tuple[np.ndarray, float]:
    """

    Fits the components over the pixels of a region, keeping every centre inside it, and returns them with
    their total chi2.

    """
    v_bounds = np.tile([pixels.start * spectrum.dv, pixels.stop * spectrum.dv], (len(components), 1))
    fit, _ = fit_components(spectrum=spectrum, components=components, pixels=pixels, v_bounds=v_bounds)
    return fit, total_chisq(spectrum=spectrum, components=fit, pixels=pixels)


def add_component(*, spectrum: SpectrumToFit, components: np.ndarray, pixels: slice) -> np.ndarray:
    """

    Appends a starting guess for a new component at the deepest absorption the components do not explain.

    """
    residual = 1.0 + spectrum.flux - spectrum.model(components)
    guess = starting_guess(spectrum=spectrum, residual=residual, pixels=pixels)
    if np.any(np.isclose(components[:, 2], guess[2], rtol=0.0, atol=10e-9)):  # same params as vpfit
        guess[2] += 0.5 * spectrum.dv  # never start two components at the same centre
    return np.vstack((components, guess))


def simpler_models(components: np.ndarray) -> list[np.ndarray]:
    """

    Every model with one component fewer: each component removed, weakest first, then each pair of neighbours
    in velocity merged into one with total column and column-weighted b and v.

    """
    removed = [np.delete(components, i, axis=0) for i in np.argsort(components[:, 0])]
    merged = []
    order = np.argsort(components[:, 2])
    for i, k in zip(order[:-1], order[1:]):
        columns = 10.0 ** components[[i, k], 0]
        b, v = columns @ components[[i, k], 1:] / np.sum(columns)
        merged.append(np.vstack((np.delete(components, [i, k], axis=0), [np.log10(np.sum(columns)), b, v])))
    return removed + merged


def prune_components(*, spectrum: SpectrumToFit, components: np.ndarray, pixels: slice) -> np.ndarray:
    """

    Replaces the components by a simpler model, refitted, whenever it raises the total chi2 by less than
    DCHISQ_PER_COMPONENT, until no simpler model does.

    """
    chisq = total_chisq(spectrum=spectrum, components=components, pixels=pixels)
    changed = True
    while changed and len(components) > 0:
        changed = False
        for trial in simpler_models(components):
            trial, trial_chisq = fit_in_region(spectrum=spectrum, components=trial, pixels=pixels)
            if trial_chisq - chisq < DCHISQ_PER_COMPONENT:
                components, chisq, changed = trial, trial_chisq, True
                break
    return components


def fit_region(*, spectrum: SpectrumToFit, pixels: slice, max_components: int, chisq_factor: float) -> np.ndarray:
    """

    Fits a region with components added one at a time, each either a new line at the deepest unexplained
    absorption or a split of an existing one, whichever fits best.

    A component is kept while it lowers the total chi2 by at least DCHISQ_PER_COMPONENT and to at most
    chisq_factor of its previous value; the components are then pruned.

    """
    first = add_component(spectrum=spectrum, components=np.zeros((0, 3)), pixels=pixels)
    components, chisq = fit_in_region(spectrum=spectrum, components=first, pixels=pixels)
    while len(components) < max_components:
        candidates = [add_component(spectrum=spectrum, components=components, pixels=pixels)]
        candidates += [split_component(components=components, index=i) for i in range(len(components))]
        fits = [fit_in_region(spectrum=spectrum, components=candidate, pixels=pixels) for candidate in candidates]
        trial, trial_chisq = min(fits, key=lambda fit: fit[1])
        if not np.isfinite(trial_chisq):
            break
        if chisq - trial_chisq < DCHISQ_PER_COMPONENT or trial_chisq > chisq_factor * chisq:
            break
        components, chisq = trial, trial_chisq
    return prune_components(spectrum=spectrum, components=components, pixels=pixels)


def fit_in_box(
    *, spectrum: SpectrumToFit, components: np.ndarray, regions: np.ndarray, region_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """

    Fits all components together over the whole box, keeping each centre inside its own region, and
    returns them with their errors and total chi2.

    """
    whole_box = slice(0, spectrum.n_pixels)
    v_bounds = regions[region_index] * spectrum.dv
    fit, errors = fit_components(spectrum=spectrum, components=components, pixels=whole_box, v_bounds=v_bounds)
    return fit, errors, total_chisq(spectrum=spectrum, components=fit, pixels=whole_box)


def fit_jointly(
    *, spectrum: SpectrumToFit, components: np.ndarray, regions: np.ndarray, region_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Refits the components of all regions together, so that line wings reaching into neighbouring regions are
    accounted for, then removes components, weakest first and one at a time with a refit, whenever this raises
    the total chi2 by less than DCHISQ_PER_COMPONENT.

    Returns the components, their errors and the region each belongs to.

    """
    fit, errors, chisq = fit_in_box(
        spectrum=spectrum, components=components, regions=regions, region_index=region_index
    )
    changed = True
    while changed and len(fit) > 0:
        changed = False
        for i in np.argsort(fit[:, 0]):
            keep = np.arange(len(fit)) != i
            trial, trial_errors, trial_chisq = fit_in_box(
                spectrum=spectrum, components=fit[keep], regions=regions, region_index=region_index[keep]
            )
            if trial_chisq - chisq < DCHISQ_PER_COMPONENT:
                fit, errors, region_index, chisq, changed = trial, trial_errors, region_index[keep], trial_chisq, True
                break

    return fit, errors, region_index


def fit_sightline(
    *,
    flux: np.ndarray,
    noise: np.ndarray,
    dv: float,
    line: LineConstants,
    kernel: np.ndarray,
    log_column_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
    n_sigma: float,
    max_components: int,
    chisq_factor: float,
) -> LineList:
    """

    Fits Voigt components to a periodic spectrum: finds the absorption regions, fits each one, then refits all
    components together over the whole box,

    The spectrum is rolled to start at its least absorbed pixel so that no region crosses the box edge, and
    the results are shifted back to the original grid.

    """
    # roll the spectrum
    shift = int(np.argmax(flux))  # find pixel with highest flux
    spectrum = SpectrumToFit(
        flux=np.roll(flux, -shift),
        noise=np.roll(noise, -shift),
        dv=dv,
        line=line,
        kernel=kernel,
        log_column_bounds=log_column_bounds,
        b_bounds=b_bounds,
    )
    # find the absorption regions
    regions = find_absorption_regions(flux=spectrum.flux, noise=spectrum.noise, n_sigma=n_sigma)

    # fit each region on its own
    found, found_in = [np.zeros((0, 3))], [np.zeros(0, dtype=np.int64)]
    for i, (start, stop) in enumerate(regions):
        if stop - start < MIN_REGION_PIXELS:
            continue
        components = fit_region(
            spectrum=spectrum, pixels=slice(start, stop), max_components=max_components, chisq_factor=chisq_factor
        )
        found.append(components)
        found_in.append(np.full(len(components), i, dtype=np.int64))

    # fit all components together
    components, errors, region_index = fit_jointly(
        spectrum=spectrum, components=np.vstack(found), regions=regions, region_index=np.concatenate(found_in)
    )

    grid = {"line": line, "n_pixels": spectrum.n_pixels, "dv": dv}
    equivalent_width = np.empty(len(components))
    reduced_chisq = np.empty(len(components))

    # compute outputs for each component
    for j, ((log_column, b, v), i) in enumerate(zip(components, region_index)):
        pixels = slice(*regions[i])
        tau = component_tau(log_column=log_column, b=b, v=v, **grid)[pixels]
        equivalent_width[j] = np.sum(1.0 - np.exp(-tau)) * dv
        chisq = total_chisq(spectrum=spectrum, components=components, pixels=pixels)
        reduced_chisq[j] = chisq / (pixels.stop - pixels.start)

    # undo the roll and go back to the original grid
    components[:, 2] = (components[:, 2] + shift * dv) % (spectrum.n_pixels * dv)
    regions = regions + shift
    regions[regions[:, 0] >= spectrum.n_pixels] -= spectrum.n_pixels

    return LineList(
        components=components,
        errors=errors,
        equivalent_width=equivalent_width,
        reduced_chisq=reduced_chisq,
        region=region_index,
        regions=regions,
    )


def run_voigt_fitting(
    *, config: OctaviusConfig, results: list[tuple[Sightlines, SightlineParticles, dict[str, SightlineSpectra]]]
) -> list[dict[str, SightlineLineLists]]:
    """

    Fits Voigt component to every spectrum from run_absorption, giving one set of line lists per sightline
    set, by line name.

    """
    if config.absorption_snr is None:
        raise ValueError("Voigt fitting needs 'absorption_snr', as the fit weighs residuals by the noise.")

    start = time.perf_counter()
    lines = load_line_table()
    line_lists = [
        {name: _fit_spectra(spectra=spectra[name], transition=lines[name], config=config) for name in spectra}
        for _, _, spectra in results
    ]
    n_components = sum(int(fitted.n_components.sum()) for by_line in line_lists for fitted in by_line.values())
    logger.info(f"Fitted {n_components} Voigt components in {time.perf_counter() - start:.1f} s.")
    return line_lists


def _fit_spectra(*, spectra: SightlineSpectra, transition: Transition, config: OctaviusConfig) -> SightlineLineLists:
    """

    Fits every sightline of one set for one line, with the noise of standard deviation 1 / SNR added in
    run_absorption, and the LSF kernel that the flux was originally convolved with.

    """
    constants = line_constants(transition=transition)
    dv = float(spectra.velocity[1] - spectra.velocity[0])
    noise = np.full(len(spectra.velocity), 1.0 / config.absorption_snr)
    fitted = [
        fit_sightline(
            flux=flux,
            noise=noise,
            dv=dv,
            line=constants,
            kernel=spectra.kernel,
            log_column_bounds=tuple(config.absorption_voigt_log_column_bounds),
            b_bounds=_b_bounds(bounds=config.absorption_voigt_b_bounds, line=constants),
            n_sigma=config.absorption_voigt_n_sigma,
            max_components=config.absorption_voigt_max_components,
            chisq_factor=config.absorption_voigt_chisq_factor,
        )
        for flux in spectra.flux
    ]
    components = np.concatenate([line_list.components for line_list in fitted])
    return SightlineLineLists(
        components=components,
        errors=np.concatenate([line_list.errors for line_list in fitted]),
        equivalent_width=transition.wavelength
        / C_KMS
        * np.concatenate([line_list.equivalent_width for line_list in fitted]),
        reduced_chisq=np.concatenate([line_list.reduced_chisq for line_list in fitted]),
        saturated=saturated_components(
            components=components,
            line=constants,
            n_pixels=len(spectra.velocity),
            dv=dv,
            kernel=spectra.kernel,
            tau_threshold=config.absorption_voigt_saturation_tau,
        ),
        region=np.concatenate([line_list.region for line_list in fitted]),
        regions=np.concatenate([line_list.regions for line_list in fitted]),
        n_components=np.array([len(line_list.components) for line_list in fitted], dtype=np.int64),
        n_regions=np.array([len(line_list.regions) for line_list in fitted], dtype=np.int64),
    )


def _b_bounds(*, bounds: list[float], line: LineConstants) -> tuple[float, float]:
    """

    Bounds on b for a line, if lower bound is set as default of 0, it is replaced by B_FLOOR_FRACTION of
    the thermal b at B_FLOOR_TEMPERATURE (0.25, and 10e4 K by default, respectively).

    """
    if bounds[0] == 0.0:
        return (B_FLOOR_FRACTION * line.b_0 * math.sqrt(B_FLOOR_TEMPERATURE), float(bounds[1]))
    return (float(bounds[0]), float(bounds[1]))
