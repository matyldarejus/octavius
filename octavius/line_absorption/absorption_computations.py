"""

Optical depth (i.e. tau) computations along sightlines.

A gas particles at impact parameter R from a sightline contributes a column
N_ion * F(R / H) / H^2, where H is its kernel support radius and F is the 3D
kernel integrated along the line of sight. F is tabulated once per kernel and
linearly interpolated.

"""

# default libraries
import math

# other packages
import numpy as np
from numba import njit, prange

# internal imports
from .line_profiles import gaussian_bin_integral, voigt_bin_integral

N_KERNEL_TABLE = 4096  # table intervals over s = R / H in range [0, 1]
N_LOS_STEPS = 128  # Simpson sub-intervals along the line of sight

N_DOPPLER_WIDTHS = 6.0  # Gaussian core window
TAU_WING_FLOOR = 1.0e-6  # value below which particle's wing tau must fall down to to exclude it


@njit(cache=True)
def kernel_3d(q, kernel_id):
    """

    3D kernel at distance from the particle centre, q = r / H (H = support radius), normalised s.t. kernel_3d(r / H) / H^3 has unit
    volume integral. Zero for q >= 1.

    Equations taken from Dehnen & Aly (2012) (doi: 10.1111/j.1365-2966.2012.21439.x), Table 1:
        - 0: cubic spline:
            (1 - r)^3 - 4(1/2 - r)^3, C = 16 / pi
        - 2: Wendland C2:
            (1 - r)^4 * (1 + 4r), C = 21 / 2pi

    """
    if q >= 1.0:
        return 0.0
    if kernel_id == 0:
        if q < 0.5:  # full equation
            return 8.0 / math.pi * (1.0 - 6.0 * q * q + 6.0 * q * q * q)
        return 16.0 / math.pi * (1.0 - q) ** 3  # drop the second term as its -ve, hence zero
    if kernel_id == 2:
        return 21.0 / (2.0 * math.pi) * (1.0 - q) ** 4 * (1.0 + 4.0 * q)
    raise ValueError("Unknown kernel_id.")


@njit(cache=True)
def build_projected_kernel_table(kernel_id, n_table=N_KERNEL_TABLE, n_steps=N_LOS_STEPS):
    """

    Tabulates F(s) = 2 * integral_0^sqrt(1 - l^2) kernel_3d(sqrt(s^2 + l^2)) dl at s = i / n_table,
    for i = [0, n_table], using Simpson's rule along l, distance along the sightline from the point of
    closest approach.

    """
    if n_steps % 2 != 0:
        raise ValueError("n_steps must be even for Simpson's rule.")
    table = np.zeros(n_table + 1)
    for i in range(n_table):
        s = i / n_table
        dx_los = math.sqrt(1.0 - s * s) / n_steps  # Simpson step
        total = kernel_3d(s, kernel_id)
        for k in range(1, n_steps):
            x_los = k * dx_los
            # weight 4 for odd k, 2 for even k
            total += (4.0 if k % 2 == 1 else 2.0) * kernel_3d(math.sqrt(s * s + x_los * x_los), kernel_id)
        table[i] = 2.0 * total * dx_los / 3.0
    return table


@njit(cache=True)
def projected_kernel(s, table):
    """

    Projected kernel F at s = R / H >= by linear interpolation in the table.

    Zero for s >= 1 as the sightline misses the particle.

    """
    if s >= 1.0:
        return 0.0
    x = s * (table.shape[0] - 1)
    i = int(x)
    frac = x - i
    return (1.0 - frac) * table[i] + frac * table[i + 1]  # do a weighted average between the two entries


def velocity_grid(*, vbox: float, pixel_kms: float) -> tuple[int, float]:
    """

    Pixel grid spanning the box's Hubble velocity.

    n_bins pixel of width dv = vbox / n_bins, with dv being the closest to the pixel_kms
    that tites the box exactly for periodic wrapping.
    Therefore, one pixel k covers [k dv, (k + 1) dv].

    """
    n_bins = max(1, round(vbox / pixel_kms))
    return n_bins, vbox / n_bins


def line_of_sight_velocities(
    *, x_los: np.ndarray, v_pec_los: np.ndarray, scale_factor: float, hubble: float, vbox: float
) -> np.ndarray:
    """

    Line-centre velocity of each particle: LOS peculiar velocity plus the Hubble flow at its position,
    wrapped into [0, vbox]. Same as in Sightlines.gal_velocity_pos.


    """
    return np.mod(v_pec_los + hubble * scale_factor * x_los, vbox)


@njit(cache=True)
def profile_half_width(column, b, xsec, gamma):
    """

    Half-width of the velocity window a particle's line is deposited over: the Gaussian core
    (+/- N * b), or for damped lines further out, to where the Lorentzian wind
    tau ~ xsec * N * gamma / (pi * dv^2) drops to TAU_WING_FLOOR.

    Without setting TAU_WING_FLOOR > 0, the alternative is to let every damped line spread
    over the whole box. This approach sets a window of velocity width 2W around every line centre,
    where W = sqrt((xsec * N * gamma) / (pi * TAU_WING_FLOOR) derived using the Voigt profile and
    setting velocity offset to be far beyond the core and the width to be much larger than the natural
    width.

    """
    core = N_DOPPLER_WIDTHS * b
    if gamma <= 0.0:
        return core
    return max(core, math.sqrt(xsec * column * gamma / (math.pi * TAU_WING_FLOOR)))


@njit(cache=True)
def _deposit_line(tau_row, weighted_rows, values, column, v, b, gamma, dv, half_width):
    """

    Adds one particle's column, spread over the pixels by its bin-integrated line profile, to one sightline:
    tau_row[k] += column * (fraction of the profile in pixel k), and the same times each of the values to
    weighted_rows[:, k]. Pixel indices wrap periodically; the window covers at most one box.

    """
    n_bins = tau_row.shape[0]
    half_width = min(half_width, 0.5 * n_bins * dv)  # nearest image only: offsets within +- vbox / 2
    i_lo = math.floor((v - half_width) / dv)
    i_hi = min(math.floor((v + half_width) / dv), i_lo + n_bins - 1)

    n_sub = 2 * max(1, math.ceil(2.0 * dv / b))  # Simpson step: b / 4

    for i in range(i_lo, i_hi + 1):
        lo = i * dv - v  # pixel edges relative to the line centre
        if gamma > 0.0:
            frac = voigt_bin_integral(lo, lo + dv, b, gamma, n_sub)
        else:
            frac = gaussian_bin_integral(lo, lo + dv, b)
        k = i % n_bins  # wrap periodically
        contribution = column * frac
        tau_row[k] += contribution
        for w in range(values.shape[0]):
            weighted_rows[w, k] += values[w] * contribution


@njit(parallel=True, cache=True)
def optical_depth(
    los_x,
    los_y,
    los_offsets,
    los_members,
    plane_x,
    plane_y,
    support,
    n_ion,
    v_line,
    b_line,
    weights,
    boxsize,
    length_to_cm,
    n_bins,
    dv,
    line,
    table,
):
    """

    Optical depth along every sightline of one set, from its preselected particles (csr: the members
    of sightline i are los_members[los_offsets[i]:los_offsets[i + 1]], rows into the per-particle arrays).

    Each particle will contribute a column N = n_ion * F(s) / H*2 (where H is the support radius, s is the
    projected distance / H, i.e. minimum image in the plane), deposited in velocity by its line profile.

    tau_k = xsec / dv * sum_j N_j * (fraction of profile j in pixel x),

    so that

    sum_k tau_k dv = xsec * sum_j N_j

    Args (per particle, kpc a unless stated):

    plane_x, plane_y, support (H), n_ion (number of ions),
    v_line (km/s, line centre in [0, vbox)), b_line (km/s), weights (n_particles, n_w) quantities to
    tau-weight. length_to_cm: a * kpc in cm. line: LineConstants. table: projected kernel table.

    Returns:
    - tau: (n_los, n_bins)
    - weighted: (n_los, n_w, n_bins) tau-weighted mean of each weight quantity (0 where tau = 0)
    - member_column: (len(los_members),) column density (cm^-2) of each sightline-particle pair, for n90


    """
    n_los = los_x.shape[0]
    n_w = weights.shape[1]
    tau = np.zeros((n_los, n_bins))
    weighted = np.zeros((n_los, n_w, n_bins))
    member_column = np.zeros(los_members.shape[0])

    for los in prange(n_los):  # prevents races between sightlines
        for j in range(los_offsets[los], los_offsets[los + 1]):
            m = los_members[j]
            dx = plane_x[m] - los_x[los]
            dy = plane_y[m] - los_y[los]
            dx -= boxsize * round(dx / boxsize)  # min image
            dy -= boxsize * round(dy / boxsize)
            s = math.sqrt(dx * dx + dy * dy) / support[m]
            if s >= 1.0:  # not on the sightline
                continue
            h_cm = support[m] * length_to_cm
            column = n_ion[m] * projected_kernel(s, table) / (h_cm * h_cm)
            member_column[j] = column
            if column <= 0.0:
                continue
            half_width = profile_half_width(column, b_line[m], line.xsec, line.gamma)
            _deposit_line(tau[los], weighted[los], weights[m], column, v_line[m], b_line[m], line.gamma, dv, half_width)

        for k in range(n_bins):
            if tau[los, k] > 0.0:
                for w in range(n_w):
                    weighted[los, w, k] /= tau[los, k]
                tau[los, k] *= line.xsec / dv

    return tau, weighted, member_column
