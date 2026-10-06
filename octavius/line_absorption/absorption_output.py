"""

Writes line absorption outputs which belong outside of the Octavius catalogue, inc. galaxy selection, spectra etc.

Currently just a stand-in as no spectra are computed as of yet.

"""

# default packages
from dataclasses import fields
from pathlib import Path

# other packages
import h5py

from ..log import get_logger

# internal imports
from .absorption_helpers import GalaxySample, SamplingParams

logger = get_logger()


def selection_file_path(*, snapshot_path: Path, output_dir: Path) -> Path:
    """

    Returns the selection output file path.

    """
    return output_dir / f"absorption_selection_{snapshot_path.stem}.hdf5"


def write_selection(
    *, path: Path, sample: GalaxySample, params: SamplingParams, redshift: float, catalogue_path: Path
) -> None:
    """

    Writes the galaxy sample selection and the parameters used to produce it.

    """
    with h5py.File(path, "w") as f:
        f.create_dataset("selected", data=sample.selected)
        f.create_dataset("eligible", data=sample.eligible)
        f.create_dataset("indices", data=sample.indices)
        f.create_dataset("ssfr_class", data=sample.ssfr_class)
        f.create_dataset("bin_idx", data=sample.bin_idx)
        f.create_dataset("bin_total", data=sample.bin_total)
        f.create_dataset("bin_labels", data=list(sample.bin_labels), dtype=h5py.string_dtype())

        for field in fields(params):  # write the parameter set and essential info
            f.attrs[field.name] = getattr(params, field.name)
        f.attrs["redshift"] = redshift
        f.attrs["catalogue"] = str(catalogue_path)

    logger.info(f"Wrote galaxy sample ({sample.n_selected} galaxies) to {path}.")
