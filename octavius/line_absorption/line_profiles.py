"""

Line data and normalised line profiles.

    - Transitions are read from line_data.csv
    - Profiles are normalised to unit integral over velocity:
        1. Gaussian (thermal + turbulent broadening, Doppler parameter b).
        2. Lorentzian (natural broadening, gamma).
        3. Voigt - done via the Tepper-Garcia (2006) approximation to the Voigt-Hjerting function.
    - Bin integrals give the fraction of a profile falling in the velocity bin, which the optical-depth
    synthesis adds up per particle.

TODO next: Decide how many steps to use for voigt profile integral.

"""

# default libraries
import csv
import math
from pathlib import Path

# other packages
import numpy as np
from astropy import units as u
from astropy.constants import codata2022 as codata
from numba import njit

# internal imports
from .absorption_helpers import LineConstants, Transition

LINE_DATA_PATH = Path(__file__).parent / "line_data.csv"
SQRT_PI = math.sqrt(math.pi)
SMALL_X_SQ = 1.0e-4


def load_line_table(*, path: Path = LINE_DATA_PATH) -> dict[str, Transition]:
    """

    Reads the line table.

    """
    with path.open() as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return {
            row["name"]: Transition(
                name=row["name"],
                ion=row["ion"],
                element=row["element"],
                wavelength=float(row["wavelength_angstrom"]),
                f=float(row["f"]),
                atomic_weight=float(row["atomic_weight_u"]),
                A_ki=float(row["A_ki_per_s"]) if row["A_ki_per_s"] else 0.0,
            )
            for row in rows
        }


def line_constants(*, transition: Transition) -> LineConstants:
    """

    Converts a transition's atomic data into the constants that the synthesis kernels use.

    Computed once.

    These were sourced from Pygad's absorption_spectra.py file -- TODO: Cross-reference

    """
    wavelength = transition.wavelength * u.AA
    ion_mass = transition.atomic_weight * u.u

    xsec = transition.f * codata.e.si**2 / (4.0 * codata.eps0 * codata.m_e * codata.c) * wavelength
    b_0 = np.sqrt(2.0 * codata.k_B * (1.0 * u.K) / ion_mass)
    gamma = transition.A_ki / u.s * wavelength / (4.0 * np.pi)

    return LineConstants(
        xsec=float(xsec.to(u.cm**2 * u.km / u.s).value),
        b_0=float(b_0.to(u.km / u.s).value),
        gamma=float(gamma.to(u.km / u.s).value),
    )


@njit(cache=True)
def gaussian(v, b):
    """

    Gaussian profie with Doppler parameter b (= sqrt(2) sigma), unit integral over v.

    Returns:
    - phi: profile value (1 / velocity units)
    """
    return math.exp(-((v / b) ** 2)) / (b * SQRT_PI)


@njit(cache=True)
def lorentzian(v, gamma):
    """

    Lorentzian profile with half width at half maximum gamma, unit integral over v.

    Returns:
        - phi: profile value (1 / velocity units)
    """
    return gamma / (math.pi * (v * v + gamma * gamma))


@njit(cache=True)
def voigt_hjerting(a, x):
    """

    Tepper-Garcia (2006) approximation to the Voigt-Hjerting function H(a, x), with a = gamma / b and
    x = v / b.

    Near x = 0 the formula's bracket / x^2 cancels badly - its series limit 2 - 4 x^2 is used there.

    Returns:
    - H: H(a, x)
    """
    x_sq = x * x
    h0 = math.exp(-x_sq)
    if x_sq < SMALL_X_SQ:
        return h0 - (a / SQRT_PI) * (2.0 - 4.0 * x_sq)
    q = 1.5 / x_sq
    return h0 - (a / SQRT_PI) / x_sq * (h0 * h0 * (4.0 * x_sq * x_sq + 7.0 * x_sq + 4.0 + q) - q - 1.0)


@njit(cache=True)
def voigt(v, b, gamma):
    """

    Voigt profile (Gaussian with Doppler parameter b convolved with a Lorentzian of
    half-width half-maximum gamma), unit integral over v.

    For gamma = 0, we get the Gaussian.

    """
    if gamma <= 0.0:
        return gaussian(v, b)
    return voigt_hjerting(gamma / b, v / b) / (b * SQRT_PI)


@njit(cache=True)
def gaussian_bin_integral(v0, v1, b):
    """

    Exact integral of the Gaussian profile over [v0, v1] (relative to the line centre).

    """
    return 0.5 * (math.erf(v1 / b) - math.erf(v0 / b))


@njit(cache=True)
def voigt_bin_integral(v0, v1, b, gamma, n_steps):
    """

    Integral of the Voigt profile over [v0, v1] using Simpson's rule with n_steps sub-intervals.

    No elementary integral function, so analytic solution is needed.

    """
    h = (v1 - v0) / n_steps
    total = voigt(v0, b, gamma) + voigt(v1, b, gamma)
    for k in range(1, n_steps):
        total += (4.0 if k % 2 == 1 else 2.0) * voigt(v0 + k * h, b, gamma)
    return total * h / 3.0
