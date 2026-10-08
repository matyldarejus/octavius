"""

Handles sightline geometry and particle preselection for the line absorption module.


For the gas pre-selection, the general idea is:

    - For every sightline we need the gas particles whose kernel the sightline passes
        through - these particles contribute to absorption
    - We reuse the logic of photometry: the spatial hashing there puts gas in the
        cells and queries with stars; here, we put the sightlines in the cells and
        query with gas particles instead.
    - So, for a particle:
        - Find the range of cells the kernel circle overlaps.
        - Count the number of sightlines in each of the cells.
        - For particles overlapping cells with sightlines, for each sightline,
            compute dx^2 + dy^2 < h^2: any sightline within passes.
        - If the circle sticks out past the edge of the box, we repeat above two steps with the
            particle shited by boxsize.
        - Masked particles (e.g. winds) get support radius (defined as 'support') = 0 and return immediately.
    - Particles are processed in parallel in two passes: first each particle counts the sightlines in overlapping
        cells, then it writes them into its own slice of the output, so threads never
        write to the same place.
    - The matches from every stream gas chunk are merged into one list per sightline.

"""

# other packages
import numpy as np
from numba import njit, prange

# internal imports
from ..photometry.photometry_computations import build_dust_cell_list
from .absorption_helpers import SightlineGrid, SightlineParams, SightlineParticles, Sightlines, plane_axes


def build_sightlines(
    *,
    entry_idx: np.ndarray,
    galaxy_idx: np.ndarray,
    centre: np.ndarray,
    velocity: np.ndarray,
    ang_mom: np.ndarray,
    r200c: np.ndarray,
    params: SightlineParams,
    los_axis: int,
    boxsize: float,
    scale_factor: float,
    hubble: float,
) -> Sightlines:
    """

    Places sightlines around each selected galaxy at every impact parameter x azimuth, parallel to the LOS axis.

    Azimuths are in the box frame (phi = 2 pi k / n_azimuth from the first plane axis); a zero impact parameter
    gives a single sightline through the centre. Positions are wrapped into the periodic box.

    """
    los = los_axis
    first, second = plane_axes(los_axis=los)
    n_sel = len(galaxy_idx)

    # the (impact parameter, azimuth) pattern every galaxy receives
    azimuths = 2.0 * np.pi * np.arange(params.n_azimuth) / params.n_azimuth  # in rads
    pattern_param = np.concatenate(
        [np.zeros(1) if p == 0.0 else np.full(params.n_azimuth, p) for p in params.impact_parameters]
    )
    pattern_phi = np.concatenate([azimuths[:1] if p == 0.0 else azimuths for p in params.impact_parameters])
    n_pattern = len(pattern_param)

    # galaxy-major ordering: galaxy, then impact parameter, then azimuth
    owner = np.repeat(np.arange(n_sel), n_pattern)
    impact_param = np.tile(pattern_param, n_sel)
    azimuth = np.tile(pattern_phi, n_sel)  # in rads

    if params.impact_in_r200c:
        impact = impact_param * r200c[owner]
    else:
        impact = impact_param / scale_factor  # convert physical kpc to kpc a

    offset_dir = np.column_stack((np.cos(azimuth), np.sin(azimuth)))  # unit vectors in the plane
    pos = centre[owner][:, [first, second]] + impact[:, None] * offset_dir
    pos = np.mod(pos, boxsize)

    vbox = hubble * scale_factor * boxsize  # km/s
    gal_velocity_pos = velocity[owner, los] + hubble * scale_factor * centre[owner, los]  # km/s
    gal_velocity_pos = np.mod(gal_velocity_pos, vbox)

    azimuth_disc, inclination = _disc_angles(
        ang_mom=ang_mom[owner], offset_dir=offset_dir, impact=impact, los=los, first=first, second=second
    )

    return Sightlines(
        pos=pos,
        entry_idx=np.asarray(entry_idx, dtype=np.int64)[owner],
        galaxy_idx=np.asarray(galaxy_idx, dtype=np.int64)[owner],
        impact_param=impact_param,
        impact=impact,
        azimuth=azimuth,
        azimuth_disc=azimuth_disc,
        inclination=inclination,
        gal_velocity_pos=gal_velocity_pos,
        los_axis=los,
        boxsize=float(boxsize),
        vbox=float(vbox),
    )


def build_random_sightlines(
    *, n_los: int, los_axis: int, boxsize: float, scale_factor: float, hubble: float, seed: int
):
    """

    Places n_los random sightlines parallel to the LOS axis at positions drawn uniformly over the box.

    """
    rng = np.random.default_rng(seed)

    # this is just across the whole box, so fill the remaining fields with -1's and Nan's
    no_galaxy = np.full(n_los, -1, dtype=np.int64)
    undefined = np.full(n_los, np.nan)

    return Sightlines(
        pos=rng.uniform(0.0, boxsize, size=(n_los, 2)),
        entry_idx=no_galaxy,
        galaxy_idx=no_galaxy.copy(),
        impact_param=undefined,
        impact=undefined.copy(),
        azimuth=undefined.copy(),
        azimuth_disc=undefined.copy(),
        inclination=undefined.copy(),
        gal_velocity_pos=undefined.copy(),
        los_axis=los_axis,
        boxsize=float(boxsize),
        vbox=float(hubble * scale_factor * boxsize),
    )


def _disc_angles(
    *, ang_mom: np.ndarray, offset_dir: np.ndarray, impact: np.ndarray, los: int, first: int, second: int
) -> tuple[np.ndarray, np.ndarray]:
    """

    Disc-relative angles per sightline, relative to the angular momentum axis.

    The projected major axis is perpendicular to the projected angular
    momentum; azimuth_disc = 0 along the major axis, pi/2 along the minor axis.

    """
    L_norm = np.linalg.norm(ang_mom, axis=1)
    L_plane = ang_mom[:, [first, second]]
    L_plane_norm = np.linalg.norm(L_plane, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        inclination = np.arccos(np.clip(np.abs(ang_mom[:, los]) / L_norm, 0.0, 1.0))
        major = np.column_stack((-L_plane[:, 1], L_plane[:, 0])) / L_plane_norm[:, None]  # (n_los, 2) unit
        cos_to_major = np.abs(np.sum(offset_dir * major, axis=1))
        azimuth_disc = np.arccos(np.clip(cos_to_major, 0.0, 1.0))

    inclination[L_norm == 0.0] = np.nan
    azimuth_disc[(L_plane_norm == 0.0) | (impact == 0.0)] = np.nan
    return azimuth_disc, inclination


def build_sightline_grid(*, sightlines: Sightlines, cell_width: float) -> SightlineGrid:
    """

    Bins one set's sightline positions into a 2D cell list using photometry's build_dust_cell_list.

    """
    first, second = plane_axes(los_axis=sightlines.los_axis)
    pos_3d = np.zeros((sightlines.n_los, 3))  # build_dust_cell_list takes in 3D positions
    pos_3d[:, first] = sightlines.pos[:, 0]
    pos_3d[:, second] = sightlines.pos[:, 1]

    sort_order, cell_offsets, n_cx, n_cy, origin_x, origin_y, inv_width = build_dust_cell_list(
        gas_pos=pos_3d,
        smoothing_lengths=np.full(sightlines.n_los, cell_width),
        ax0=first,
        ax1=second,
    )

    return SightlineGrid(
        los_x=sightlines.pos[:, 0].copy(),
        los_y=sightlines.pos[:, 1].copy(),
        sort_order=sort_order,
        cell_offsets=cell_offsets,
        n_cells_x=n_cx,
        n_cells_y=n_cy,
        origin_x=origin_x,
        origin_y=origin_y,
        inv_cell_width=inv_width,
    )


def find_overlaps(*, part_pos: np.ndarray, support: np.ndarray, grid: SightlineGrid, los_axis: int, boxsize: float):
    """

    Finds every (particle, sightline) pair where the sightline passes within the particle's kernel support radius.

    Particles with support = 0 (e.g. wind particles) match nothing.

    """
    if len(grid.los_x) == 0 or len(part_pos) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    first, second = plane_axes(los_axis=los_axis)
    part_x = np.mod(part_pos[:, first], boxsize)
    part_y = np.mod(part_pos[:, second], boxsize)
    support = np.ascontiguousarray(support, dtype=np.float64)

    counts = _count_overlaps(part_x, part_y, support, grid, boxsize)  # count sightlines overlapping
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    los_idx = np.empty(starts[-1], dtype=np.int64)
    _fill_overlaps(part_x, part_y, support, grid, boxsize, starts, los_idx)

    rows = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return rows, los_idx


def assemble_sightline_particles(
    *, particle_idx_chunks: list[np.ndarray], los_idx_chunks: list[np.ndarray], n_los: int
) -> SightlineParticles:
    """

    Merges per-chunk pairs into the per-sightline csr.

    """
    if not particle_idx_chunks:
        particle_idx_chunks, los_idx_chunks = [np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)]

    global_idx = np.concatenate(particle_idx_chunks)
    los_idx = np.concatenate(los_idx_chunks)

    particle_idx, rows = np.unique(global_idx, return_inverse=True)  # sort
    order = np.argsort(los_idx, kind="stable")

    los_offsets = np.zeros(n_los + 1, dtype=np.int64)
    np.cumsum(np.bincount(los_idx, minlength=n_los), out=los_offsets[1:])

    return SightlineParticles(
        particle_idx=particle_idx.astype(np.int64), los_offsets=los_offsets, los_members=rows[order].astype(np.int64)
    )


@njit(cache=True)
def _particle_overlaps(x, y, h, grid, boxsize, out, start, write):
    """

    Visits the cells covered by one particle's kernel, counting, and if write recording from
    out[start], the sightline within h.

    """
    n_found = 0
    # get rid of masked particles
    if h <= 0.0:
        return n_found
    h_sq = h * h

    # find periodic image in x
    for shift_x in (0.0, boxsize, -boxsize):
        if shift_x > 0.0 and x - h >= 0.0:
            continue  # no low edge crossing
        if shift_x < 0.0 and x + h <= boxsize:
            continue  # no high edge crossing
        xi = x + shift_x  # x of the image
        # the impact circle spans [xi - h, xi + h]
        ix0 = max(int(np.floor((xi - h - grid.origin_x) * grid.inv_cell_width)), 0)
        ix1 = min(int(np.floor((xi + h - grid.origin_x) * grid.inv_cell_width)), grid.n_cells_x - 1)
        if ix0 > ix1:  # the circle lies outside the grid
            continue

        # repeat in other axis
        for shift_y in (0.0, boxsize, -boxsize):
            if shift_y > 0.0 and y - h >= 0.0:
                continue
            if shift_y < 0.0 and y + h <= boxsize:
                continue
            yi = y + shift_y
            iy0 = max(int(np.floor((yi - h - grid.origin_y) * grid.inv_cell_width)), 0)
            iy1 = min(int(np.floor((yi + h - grid.origin_y) * grid.inv_cell_width)), grid.n_cells_y - 1)
            if iy0 > iy1:
                continue

            # now look at every cell in the whole rectangle
            for cx in range(ix0, ix1 + 1):
                for cy in range(iy0, iy1 + 1):
                    cell = cx * grid.n_cells_y + cy
                    for k in range(grid.cell_offsets[cell], grid.cell_offsets[cell + 1]):
                        s = grid.sort_order[k]
                        dx = grid.los_x[s] - xi
                        dy = grid.los_y[s] - yi
                        if dx * dx + dy * dy < h_sq:  # inside the kernel's support radius, i.e. we have an overlap
                            if write:
                                out[start + n_found] = s
                            n_found += 1

    return n_found


@njit(cache=True, parallel=True)
def _count_overlaps(part_x, part_y, support, grid, boxsize):
    """

    Per particle, performs a cell search, finding sightlines within cells, counts them,
    and write the counts into a slot.

    This allows to skip particles which do not cross any sightlines and allows for easy parallelisation.

    """
    n = len(part_x)
    counts = np.zeros(n, dtype=np.int64)
    dummy = np.empty(0, dtype=np.int64)
    for i in prange(n):
        counts[i] = _particle_overlaps(part_x[i], part_y[i], support[i], grid, boxsize, dummy, 0, False)
    return counts


@njit(cache=True, parallel=True)
def _fill_overlaps(part_x, part_y, support, grid, boxsize, starts, out):
    """

    For particles which have non-zero sightline counts, search for sightlines again, and
    write their positions into its own slice out[starts[i]:starts[i+1]].

    """
    for i in prange(len(part_x)):
        _particle_overlaps(part_x[i], part_y[i], support[i], grid, boxsize, out, starts[i], True)
