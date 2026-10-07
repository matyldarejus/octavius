"""

Handles sightline geometry and particle preselection for the line absorption module.

"""

# other packages
import numpy as np

# internal imports
from .absorption_helpers import SightlineParams, Sightlines, plane_axes


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
