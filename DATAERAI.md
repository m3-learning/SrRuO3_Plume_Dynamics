# SrRuO3 study preservation with Dataerai

Start with [0_Dataerai_Preservation.ipynb](notebooks/0_Dataerai_Preservation.ipynb).
The catalogue in [preservation/manifest.json](preservation/manifest.json) records
Dataerai asset IDs, exact content versions, file sizes and SHA-256 hashes. It
covers all available source data, code, notebooks with their saved outputs,
figures, videos, manuscript sources and the existing book build.

## Install and authenticate

Use Python 3.10+ and install the integration in the same environment as Jupyter:

```sh
python -m pip install -r requirements-dataerai.txt
dataerai auth login --server https://beta.dataerai.com
dataerai auth status
# On a remote machine: dataerai auth login --device --server https://beta.dataerai.com
```

The pinned beta CLI/SDK versions were used to validate this integration.
Install the original scientific requirements separately when running the
analysis notebooks: `python -m pip install -r requirements.txt`.
Authentication uses the CLI credential store; no credentials go in notebooks,
the catalogue, or Git. Access to the preservation collection is required.
Publishing does not make the collection public or change sharing permissions.

## Preserve the study once

The latest upstream commit removed `data/`. A retained earlier checkout can
supply the source folder, or recover the tracked data from its Git commit:

```sh
mkdir -p .dataerai/source
git archive 11a1b18 data | tar -x -C .dataerai/source
export SRRUO3_DATA_SOURCE="$PWD/.dataerai/source/data"
```

Notebook 0 inventories that folder and the current repository, builds archives
by modality, uploads them to **SrRuO3 Plume Dynamics / Preservation**, downloads
each exact content version, verifies its bytes, and writes a portable catalogue.
Each archive has a per-file inventory. Filenames and directory layout are
retained even when different experiments use the same basename. Repeated
publication reuses already verified, unchanged bundles. Changed bytes produce
new identities. Keep and commit the updated `preservation/manifest.json`.

Already-published catalogues work on fresh clones without the original local
data. Notebook 0 can reuse their pinned IDs and register the later raw inputs.
Local archives, downloaded files and run workspaces are ignored by Git.

## Add the six raw recording DIDs later

The six original HDF5 plume recordings will be uploaded separately. In notebook
0, fill `RECORDING_DIDS` with the exact expected filename and its Dataerai DID.
Each asset must contain one HDF5 recording under that filename. The helper
resolves the DID through the authenticated SDK, downloads it, and records its
actual content version, checksum and size. It never guesses a DID or substitutes
sample data. Partial registration can be resumed; all six must be registered
before notebooks 4 and 5 can run from top to bottom. Alternatively, set
`SRRUO3_RAW_SOURCE` to a directory containing all six original filenames when
building the initial preservation archives.

Notebooks 2, 3, 6 and 7 use the available microscopy, structural and processed
plume datasets. Abstract and manuscript notebooks retrieve study data and
figures. The two notebooks that need raw recordings stop at setup with an
explicit missing-input message until the DIDs are supplied.

## Run any existing notebook

Run from the top. The added setup cell finds the repository from either its root
or a notebook folder. It downloads the required bundles from Dataerai (or uses
a previously downloaded archive only after checking its hash), validates every
file, and creates a separate workspace under `.dataerai/runs/<run-id>/`.
Original `../data/` and `../figures/` paths point to the restored copies.
Scientific cells are unchanged except for the `%%dataerai` capture line.
Previously saved outputs and figure files are retained, not regenerated in Git.

Every code cell records source, streams, execution status, displayed images,
rich output, assigned arrays and tables, and saved files. Saved PNG/SVG/TIFF
figures, MP4 videos, and changed CSV/NPY files are uploaded without re-encoding.
The graph links the run to notebook source and exact input versions, cells in
execution order, and outputs to the cells that generated them. The final cell
uploads a run summary and restores the starting directory. Capture failures
raise an error and leave recovery files in the local spool.

Run assets default to **SrRuO3 Plume Dynamics / Notebook runs**. To select an
existing destination, set `DATAERAI_OWNER_TYPE=project`, `DATAERAI_OWNER_ID` and
optionally `DATAERAI_COLLECTION_ID` to the relevant UUIDs before Jupyter starts.
Input access and output ownership are separate: changing the output project
does not change the pinned input catalogue.

Saved results are described in [preservation/key_results.json](preservation/key_results.json),
including exact source notebook/cell references and verbatim numerical output.
These are historical reported results, not claims that all analyses were rerun.
The incident-velocity windows differ between notebooks 5/6 and notebook 7; the
original windows and calculations are preserved. Future results captured by
`%%dataerai` carry their run and cell identity.

## Validation

```sh
python scripts/validate_dataerai_notebooks.py
python -m pytest tests -q
# Optional, authenticated round trip (creates validation records in Dataerai):
python scripts/verify_dataerai_roundtrip.py
```

The preservation validator compares every original cell, output, execution
count, attachment and cell metadata against upstream commit `d953252`; it also
checks all 70 original figure files byte for byte. Tests cover integrity,
version-pinned downloads, safe archive extraction, missing inputs, DID
registration, cell provenance, saved files, and errors. Complete scientific
re-execution requires the raw HDF5 recordings and the original analysis
packages; preserving historical outputs does not imply a successful rerun.

The cell recorder is adapted from the QIS Summer School integration. Its
repository-specific additions are modality bundles, reproducible input
restoration, saved-file capture, original-output validation, and delayed raw
recording DID registration.
