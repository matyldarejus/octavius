"""

Post-catalogue mock absorption spectra around the galaxies of the most massive halo, with and without Voigt fits.

"""

from pathlib import Path
import numpy as np
import octavius as oc

catalogue_path = Path("/path/to/catalogue.hdf5")

# instantiate necessary data structures
config = oc.OctaviusConfig(...)  # type parameters or load from YAML (absorption_lines, absorption_snr, etc.)
catalogue = oc.load_catalogue(catalogue_path)

# get galaxies of most massive halo
halo_masses = catalogue.haloes.get_dataset("mass_total")
gal_indices = catalogue.haloes.get_galaxies(halo_index=np.argmax(halo_masses))

# build analyser and make spectra around those galaxies; also written to octavius_<snapshot>_absorption.hdf5
analyser = oc.build_analyser(catalogue=catalogue, config=config)
results = analyser.compute_absorption(group_indices=gal_indices)

for sightlines, _, spectra in results:  # one set per line-of-sight axis
    velocity = spectra["OVI1031"].velocity  # km/s
    flux = spectra["OVI1031"].flux  # (n_los, n_pixels), with noise and the LSF

# fit Voigt components as well, written to another file
analyser.update_config(absorption_voigt_fit=True)
analyser.compute_absorption(group_indices=gal_indices, path=Path("/path/to/fitted_absorption.hdf5"))
