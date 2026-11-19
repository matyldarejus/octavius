"""

Writes the mock absorption spectra from run_absorption to their own HDF5 file, next to the catalogue.

This is done one row per sightline, with the sightline sets concatenated in X, Y, Z order. Spectra are created
at their full size and then filled one sightline set at a time.


"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management.conventions import SimulationAttributes
    from ..data_management.pipeline_management import Internals

# default libraries
from pathlib import Path

# other packages
import h5py
import numpy as np

# internal imports
from ..data_management.conventions import OctaviusConfig
from ..data_management.pack_catalogue_data import write_catalogue_headers
from .absorption_helpers import SightlineParticles, Sightlines, SightlineSpectra, SightlineLineLists
from .line_profiles import load_line_table

ABSORPTION_FORMAT_VERSION = 1
SPECTRUM_DTYPE = np.float32  # rounding error should be far below the typical noise
SIGHTLINE_FIELDS = {  # fields to be written into the absorption catalogue
    "pos": ("kpc", 1, "sightline position in the plane of the sky"),
    "entry_idx": ("", 0, "row of the sample entry the sightline belongs to; -1 for none"),
    "galaxy_idx": ("", 0, "catalogue galaxy index; -1 for none"),
    "impact_param": ("absorption_impact_units", 0, "impact_parameter"),
    "impact": ("kpc", 1, "impact parameter"),
    "azimuth": ("rad", 0, "azimuth from the first plane axis"),
    "azimuth_disc": ("rad", 0, "azimuth from the projected major axis; NaN if face-on or b = 0"),
    "inclination": ("rad", 0, "angle between L and the LOS axis (0 = face-on); NaN if L = 0"),
    "gal_velocity_pos": ("km/s", 0, "galaxy LOS velocity including the Hubble flow"),
}
WEIGHTED_UNITS = {"temperature": "K", "rho": "g cm^-3", "metallicity": "", "v_pec": "km/s"}


def absorption_output_path(*, snapshot_path: Path, output_dir: Path) -> Path:
    """

    Path of the absorption spectra file, next to the catalogue.

    """
    return output_dir / f"octavius_{snapshot_path.stem}_absorption.hdf5"


def write_absorption_spectra(
    *,
    path: Path,
    results: list[tuple[Sightlines, SightlineParticles, dict[str, SightlineSpectra]]],
    config: OctaviusConfig,
    internals: "Internals",
    sim_attrs: "SimulationAttributes",
    n_ranks: int = 1,
    line_lists: list[dict[str, SightlineLineLists]] | None = None,
) -> None:
    """

    Writes the sightlines and their synthesised spectra into an output catalogue.

    """
    sightline_sets = [sightlines for sightlines, _, _ in results]
    set_spectra = [spectra for _, _, spectra in results]
    first_row = np.concatenate([[0], np.cumsum([s.n_los for s in sightline_sets])])
    n_total = first_row[-1]
    lines = load_line_table()

    with h5py.File(path, "w") as f:
        # write essential info
        f.attrs["absorption_format_version"] = ABSORPTION_FORMAT_VERSION
        f.attrs["vbox"] = sightline_sets[0].vbox  # in km/s

        # write sightlines
        geometry = f.create_group("sightlines")
        for field, (unit, a_exp, description) in SIGHTLINE_FIELDS.items():
            ds = geometry.create_dataset(
                field, data=np.concatenate([getattr(s, field) for s in sightline_sets]), compression=1
            )
            ds.attrs["unit"] = unit
            ds.attrs["a_exp"] = a_exp
            ds.attrs["description"] = description
        geometry.create_dataset(
            "los_axis",
            data=np.concatenate([np.full(s.n_los, s.los_axis, dtype=np.int8) for s in sightline_sets]),
            compression=1,
        )

        # write absorption line results
        for name in config.absorption_lines:
            velocity = set_spectra[0][name].velocity
            group = f.create_group(name)
            group.attrs["wavelength"] = lines[name].wavelength  # rest frame wavelength in Angstrom
            group.create_dataset("velocity", data=velocity).attrs["unit"] = "km/s"

            shape = (n_total, len(velocity))
            tau = group.create_dataset("tau", shape=shape, dtype=SPECTRUM_DTYPE, compression=1)
            flux = group.create_dataset("flux", shape=shape, dtype=SPECTRUM_DTYPE, compression=1)
            n90 = group.create_dataset("n90", shape=(n_total,), dtype=np.int64, compression=1)
            ew = {
                name: group.create_dataset(name, shape=(n_total,), dtype=SPECTRUM_DTYPE)
                for name in {"ew", "ew_intrinsic"}
            }
            for ds in ew.values():
                ds.attrs["unit"] = "Angstrom"
            weighted = {
                quantity: group.create_dataset(f"weighted/{quantity}", shape=shape, dtype=SPECTRUM_DTYPE, compression=1)
                for quantity in WEIGHTED_UNITS
            }
            for quantity, ds in weighted.items():
                ds.attrs["unit"] = WEIGHTED_UNITS[quantity]

            for k, spectra in enumerate(set_spectra):  # one row per sightline set
                rows = slice(first_row[k], first_row[k + 1])
                spectrum = spectra[name]
                tau[rows] = spectrum.tau
                flux[rows] = spectrum.flux
                n90[rows] = spectrum.n90
                for field, ds in ew.items():
                    ds[rows] = getattr(spectrum, field)
                for quantity, ds in weighted.items():
                    ds[rows] = spectrum.weighted[quantity]
            if line_lists is not None:
                _write_line_lists(
                    group=group.create_group("voigt"), line_lists=[by_line[name] for by_line in line_lists]
                )

    # write catalogue headers
    write_catalogue_headers(
        catalogue_path=path,
        snapshot_path=config.snapshot_path,
        config=config,
        internals=internals,
        sim_attrs=sim_attrs,
        n_ranks=n_ranks,
    )


def _write_line_lists(*, group: h5py.Group, line_lists: list[SightlineLineLists]) -> None:
    """

    Writes the fitted components of every sightline set one after another: sightline i owns the components
    los_offsets[i]:los_offsets[i + 1], and the regions region_offsets[i]:region_offsets[i + 1].

    """
    components = np.concatenate([line_list.components for line_list in line_lists])
    errors = np.concatenate([line_list.errors for line_list in line_lists])
    for k, (field, unit) in enumerate([("log_column", "log10 cm^-2"), ("b", "km/s"), ("v", "km/s")]):
        group.create_dataset(field, data=components[:, k]).attrs["unit"] = unit
        group.create_dataset(f"{field}_error", data=errors[:, k]).attrs["unit"] = unit

    for field, unit in [("equivalent_width", "Angstrom"), ("reduced_chisq", ""), ("region", ""), ("regions", "pixel")]:
        data = np.concatenate([getattr(line_list, field) for line_list in line_lists])
        group.create_dataset(field, data=data).attrs["unit"] = unit

    n_components = np.concatenate([line_list.n_components for line_list in line_lists])
    n_regions = np.concatenate([line_list.n_regions for line_list in line_lists])
    group.create_dataset("los_offsets", data=np.concatenate(([0], np.cumsum(n_components))))
    group.create_dataset("region_offsets", data=np.concatenate(([0], np.cumsum(n_regions))))
