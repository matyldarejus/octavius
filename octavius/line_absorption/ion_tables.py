"""

Determines the ion fraction using Cloudy ionisation tables

For optical_depth calculations, we require n_ion. This script takes a particle's
particle number density and temperature to calculation fraction of its element in
the required ionisation state




"""

# default libraries
import re
from dataclasses import dataclass
from pathlib import Path

# other packages
import numpy as np
from astropy.constants import codata2022 as codata

# internal imports
from ..log import get_logger

logger = get_logger()


# table headers abbreviate element names down to 4 letters + numeral for ion
ELEMENT_PREFIXES = {
    "Hydr": "H",
    "Heli": "He",
    "Carb": "C",
    "Nitr": "N",
    "Oxyg": "O",
    "Neon": "Ne",
    "Magn": "Mg",
    "Alum": "Al",
    "Sili": "Si",
    "Sulp": "S",
    "Phos": "P",
    "Iron": "Fe",
}
ION_NAME = re.compile(r"[A-Z][a-z]*[IVX]+")  # element name + numeral
ROMAN_SUFFIX = re.compile(r"[IVX]+$")

# set default Omega_b / Omega_m when the snapshot has no Omega_b
# reuse the constant used in Simba / Simba-c
DEFAULT_BARYON_FRACTION = 0.16

# cap f_G for attenuation in very dense gas
SELF_SHIELDING_FLOOR = 1.0e-3

# UV background file path
UVB_PATHS = {  # Gamma_HI(z) tables
    "HM12": Path(__file__).parent / "data" / "uvb_hm12.csv",
    "FG20": Path(__file__).parent / "data" / "uvb_fg20.csv",
}


@dataclass(frozen=True, slots=True)
class IonTable:
    """

    Dataclass for holding the Cloudy ionisation data table.

    """

    redshift: float
    log_nH: np.ndarray
    log_T: np.ndarray
    ions: tuple[str, ...]
    log_fraction: np.ndarray


def _parse_header(line: str) -> tuple[float, tuple[str, ...]]:
    """

    Cloudy table headers have a following format: "#Hdens Temp [ion name] redshift= z"/

    Reads such a header and returns the redshift and ion names in simple form.

    """
    body, sep, z_text = line.lstrip("#").partition("redshift=")
    tokens = body.split()
    if not sep or tokens[:2] != ["Hdens", "Temp"]:
        raise ValueError(f"Unexpected ion table header: {line[:80]!r}")

    ions = []
    for name in ION_NAME.findall(" ".join(tokens[2:])):
        state = ROMAN_SUFFIX.search(name).group()
        element = ELEMENT_PREFIXES.get(name[:4])
        if element is None:
            raise ValueError(f"Unknown element in ion table header: {name!r}")
        ions.append(element + state)
    return float(z_text), tuple(ions)


def read_ion_table(*, path: Path) -> IonTable:
    """

    Reads one Cloudy table file.


    """
    with path.open() as f:
        redshift, ions = _parse_header(f.readline())
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if data.shape[1] != 2 + len(ions):
        raise ValueError(f"{path.name}: {data.shape[1] - 2} data columns but {len(ions)} ions in the header.")

    log_nH, log_T = np.unique(data[:, 0]), np.unique(data[:, 1])
    if len(data) != len(log_nH) * len(log_T):
        raise ValueError(f"{path.name}: rows do not form a complete (n_H, T) grid.")
    grid = data.reshape(len(log_nH), len(log_T), data.shape[1])
    if not (np.all(grid[:, :, 0] == log_nH[:, None]) and np.all(grid[:, :, 1] == log_T[None, :])):
        raise ValueError(f"{path.name}: rows are not ordered n_H-major, T-minor.")

    return IonTable(redshift=redshift, log_nH=log_nH, log_T=log_T, ions=ions, log_fraction=grid[:, :, 2:].copy())


def _table_redshifts(*, table_dir: Path, pattern: str) -> dict[float, Path]:
    """

    Gets every file in table_dir whose name matches a given pattern. Here, specifically used to look up
    files with '<z>' in its name.

    """
    if pattern.count("<z>") != 1:
        raise ValueError(f"Ion table pattern must contain '<z>' exactly once: {pattern!r}.")
    name = re.compile(re.escape(pattern).replace("<z>", r"\d+"))

    tables = {}
    for path in sorted(table_dir.iterdir()):
        if name.fullmatch(path.name):
            with path.open() as f:
                redshift, _ = _parse_header(f.readline())
            tables[redshift] = path
    if not tables:
        raise FileNotFoundError(f"No ion tables matching {pattern!r} in {table_dir}.")
    return tables


def load_ion_table(*, table_dir: Path, pattern: str, redshift: float) -> IonTable:
    """

    Loads the ionisation table at the snapshot redshift. Does a linear interpolation in z between two table files if
    it falls between. If it matches one file, it is left as is.

    """
    tables = _table_redshifts(table_dir=table_dir, pattern=pattern)
    z_grid = np.array(sorted(tables))

    # make sure that redshift is found within the available tables
    exact = np.isclose(z_grid, redshift, rtol=0.0, atol=1e-5)
    if exact.any() or redshift < z_grid[0] or redshift > z_grid[-1]:
        if not exact.any():
            logger.warning(
                f"Redshift {redshift:.4f} outside the ion tables ({z_grid[0]:.3f}-{z_grid[-1]:.3f}); using the closest edge."
            )
        nearest = z_grid[np.argmin(np.abs(z_grid - redshift))]
        return read_ion_table(path=tables[nearest])

    # for interpolation, z_grid[i - 1] < redshift , z_grid[z]
    i = np.searchsorted(z_grid, redshift)  # get index to find the closest matching redshift tables
    lo, hi = read_ion_table(path=tables[z_grid[i - 1]]), read_ion_table(path=tables[z_grid[i]])

    # make sure both tables have the same ions
    if lo.ions != hi.ions or not (np.allclose(lo.log_nH, hi.log_nH) and np.allclose(lo.log_T, hi.log_T)):
        raise ValueError(f"Ion tables at z = {lo.redshift} and {hi.redshift} have different grids or ions.")

    w = (redshift - lo.redshift) / (hi.redshift - lo.redshift)
    return IonTable(
        redshift=redshift,
        log_nH=lo.log_nH,
        log_T=lo.log_T,
        ions=lo.ions,
        log_fraction=(1.0 - w) * lo.log_fraction + w * hi.log_fraction,
    )


def _locate_in_grid(grid: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    Returns lower grid index and fractional position of each x.

    x outside the grid takes the edge value.

    """
    i = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
    w = np.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0)

    return i, w


def log_ion_fraction(*, table: IonTable, ion: str, log_nH: np.ndarray, log_T: np.ndarray) -> np.ndarray:
    """

    Calculates the log ion fraction using bilinear interpolation in (log_nH, log_T).

    For values outside of the grid, returns the closest edge value.

    """
    if ion not in table.ions:
        raise KeyError(f"Ion '{ion}' is not in the ion table (available ions: {', '.join(table.ions)}).")
    values = table.log_fraction[:, :, table.ions.index(ion)]

    n_outside = np.count_nonzero(
        (log_nH < table.log_nH[0]) | (log_nH > table.log_nH[-1]) | (log_T < table.log_T[0]) | (log_T > table.log_T[-1])
    )

    if n_outside:
        logger.debug(f"{n_outside:,} particles outside the ion table; truncating to edges.")

    i, a = _locate_in_grid(table.log_nH, log_nH)
    j, b = _locate_in_grid(table.log_T, log_T)
    return (
        (1.0 - a) * (1.0 - b) * values[i, j]
        + a * (1.0 - b) * values[i + 1, j]
        + (1.0 - a) * b * values[i, j + 1]
        + a * b * values[i + 1, j + 1]
    )


def _read_uvb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """

    Reads the UV background table.

    """
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    log1pz, gamma = np.loadtxt(rows[1:], delimiter=",", unpack=True)  # skip the column header
    return log1pz, gamma


def gamma_hi(*, redshift: float, uvb: str) -> float:
    """

    HI photoionisation rate for the chosen UV background, interpolated linearly in log(1 + z).

    """
    log1pz, gamma = _read_uvb(UVB_PATHS[uvb])
    return float(np.interp(np.log10(1.0 + redshift), log1pz, gamma))


def sigma_hi(*, redshift: float, uvb: str) -> float:
    """

    Effective HI photoionisation cross-section for the chosen UV background. This reuses the fits
    for Eq. 14 of Rahmati et al. (2013). May be updated if better fit is found.

    """
    if uvb == "HM12":
        return 2.67e-18 * (1.0 + redshift) ** -0.018
    if uvb == "FG20":
        return -4.89946e-19 * (1.0 + redshift) ** (0.46765) + 3.072648e-18
    raise ValueError(f"No HI cross-section for UV background '{uvb}' (have: {', '.join(UVB_PATHS)}).")


def self_shielding_factor(
    *, n_H: np.ndarray, T: np.ndarray, redshift: float, baryon_fraction: float = DEFAULT_BARYON_FRACTION, uvb: str
) -> np.ndarray:
    """

    Fraction f_G of the UV background reaching the gas, from Rahmati et al. (2013), eqs. 13-14:

        n_H,ss = 6.73e-3 (sigma_HI / 2.49e-18)^(-2/3) (f_b / 0.17)^(-1/3) (Gamma_HI / 1e-12)^(2/3) (T / 1e4)^0.17
        f_G = 0.98 (1 + (n_H / n_H,ss)^1.64)^(-2.28) + 0.02 (1 + n_H / n_H,ss)^(-0.84)

    floored at SELF_SHIELDING_FLOOR.

    """
    n_ss = (
        6.73e-3
        * (sigma_hi(redshift=redshift, uvb=uvb) / 2.49e-18) ** (-2.0 / 3.0)
        * (baryon_fraction / 0.17) ** (-1.0 / 3.0)
        * (gamma_hi(redshift=redshift, uvb=uvb) * 1e12) ** (2.0 / 3.0)
        * (T / 1e4) ** 0.17
    )
    x = n_H / n_ss
    f_G = 0.98 * (1.0 + x**1.64) ** -2.28 + 0.02 * (1.0 + x) ** -0.84
    return np.maximum(f_G, SELF_SHIELDING_FLOOR)


def ion_fraction(
    *,
    table: IonTable,
    ion: str,
    n_H: np.ndarray,
    T: np.ndarray,
    redshift: float,
    self_shield: bool,
    baryon_fraction: float = DEFAULT_BARYON_FRACTION,
    uvb: str,
) -> np.ndarray:
    """

    Computes the ionisation fraction (f_ion) of each particle with optional correction for self-shielding.

    """
    n_lookup = (
        n_H / self_shielding_factor(n_H=n_H, T=T, redshift=redshift, baryon_fraction=baryon_fraction, uvb=uvb)
        if self_shield
        else n_H
    )
    return 10.0 ** log_ion_fraction(table=table, ion=ion, log_nH=np.log10(n_lookup), log_T=np.log10(T))


def hydrogen_number_density(*, density: np.ndarray, hydrogen_fraction: np.ndarray | float) -> np.ndarray:
    """

    Calculates n_H = X_H rho / m_p [cm^-3] from the physical gas density [g cm^-3].

    Use X_H = 1 to reproduce the formula used by pygad:

    n_H = rho / m_p

    """
    return hydrogen_fraction * density / codata.m_p.cgs.value


def ion_counts(*, fraction: np.ndarray, element_mass: np.ndarray, atomic_weight: float) -> np.ndarray:
    """

    Number of ions per particle, n_ion = f_ion * M_element / (A_m_u), with M_element in grams and A the
    atomic weight.

    """
    return fraction * element_mass / (atomic_weight * codata.u.cgs.value)
