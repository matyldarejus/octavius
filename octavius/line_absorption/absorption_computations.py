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
from numba import njit

N_KERNEL_TABLE = 4096  # table intervals over s = R / H in range [0, 1]
N_LOS_STEPS = 128  # Simpson sub-intervals along the line of sight, CHECK if optimal

# TODO: test convergence


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
