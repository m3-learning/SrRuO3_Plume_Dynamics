# Run the SrRuO3 notebooks with Dataerai

The raw inputs come from [SrRuO3 Data](https://beta.dataerai.com/p/99185d30-fe63-497c-91b4-852a02574d49).
The project contains **Base notebooks** and **Notebook runs** collections. Base
notebooks include the original seven analysis notebooks, a readiness notebook,
and a small analysis of the real G5 HDF5 recording. Download/sync the base
notebooks before opening them in Jupyter.

## Environment and login

The project root has one notebook dependency definition, `pyproject.toml`,
uploaded from [notebook-environment/pyproject.toml](notebook-environment/pyproject.toml).
Child collections inherit it. It selects Python 3.11 and pins the scientific
libraries, Dataerai SDK/CLI, and Jupyter dependencies. A complete record of the
locally validated environment is in `notebook-environment/requirements.lock.txt`.
The scientific environment uses SciPy 1.15.3 because the Materials Project
client imports interpolation APIs absent from the upstream requirements file's
SciPy 1.13.1. XRD-Learn, AFM-Learn, and Plume-Learn are pinned to exact
Git revisions because the notebooks use APIs absent from their PyPI releases.
Building the environment requires Git. AFM-Learn's legacy Igor import uses
the maintained `igor2` binary reader; this avoids removed NumPy aliases without changing the scientific cells.

In Dataerai Desktop, open **Base notebooks**, review its inherited environment,
and build it. Choose/download the notebook files, then launch JupyterLab from
that workspace. Local JupyterLab is a separate organization feature from
notebook environment indexing. If it is unavailable, use an existing local
Jupyter installation with the environment below.

For a repository checkout, with `uv` installed:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r notebook-environment/requirements.lock.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m ipykernel install --user --name srruo3 --display-name 'SrRuO3 Dataerai'
.venv/bin/dataerai auth login --server https://beta.dataerai.com --device
.venv/bin/dataerai auth status
```

Select the **SrRuO3 Dataerai** kernel. Credentials remain in the CLI credential
store. Do not place tokens in notebooks, dependency files, or Git. Ordinary
interactive login can omit `--device`.

Before any input transfer or analysis, setup verifies the beta server, refreshes
credentials through the SDK, checks the signed-in identity against the transfer
daemon, confirms raw-data read access and destination write access, and checks
the installed packages against the declared environment. An invalid login,
wrong server, missing permission, identity mismatch, or incompatible runtime
stops the notebook with an actionable error.

Notebook environment indexing must be enabled for the owning organization;
the signed-in user must also be entitled to use it in that organization. Data
read access alone does not grant this feature. Publication records the resolved
environment, exact dependency content version, and environment hash.

## Start here

1. Run `0_Dataerai_Preservation.ipynb` to check access and the pinned catalogue.
   It starts and finalizes a real provenance run without downloading the raw recordings.
2. Run `00_Dataerai_Raw_Data_Quickstart.ipynb` to download and verify the real G5
   recording (3.70 GB), inspect its datasets, and analyze one 128×128 frame window.
   It saves a table, NumPy array, rendered plot, original SVG, executed notebook,
   and run summary in Dataerai. This is an input/provenance check, not a complete
   plume-dynamics re-analysis.
3. Run an original numbered notebook from the top for its scientific analysis.
   Notebooks 4 and 5 download all six raw recordings (about 32 GB). Other
   notebooks download only their modalities; the 6/7 analyses use the historical
   processed metrics. Full-resolution analysis may require substantial memory.

Published base notebooks contain no old execution outputs. Their first cell
checks login and downloads an exact version of the repository source workspace
from Dataerai. It checks SHA-256 and archive paths before loading that code.
Each execution has a separate workspace, independent of the original files.

## Exact inputs and current upstream paths

`preservation/project-inputs.json` pins 150 accessible project files by asset ID,
content version, path, byte size, and repository BLAKE3/SHA-256 checksum. All six
recordings are present as `Plumes/G1.h5` through `Plumes/G5.h5` and
`Plumes/C-G6.h5`. Four older top-level AFM text records return 404; they are
listed under `unavailable_files` and are outside the folders used by the current
analysis notebooks. An inaccessible input required by a notebook stops setup.

Downloads always specify the pinned `content_id`. Cached bytes are checked
again before reuse; corrupt or mismatched downloads cannot enter a run.
Paths and basenames are retained, including repeated filenames in different
experiment folders. The original archives in `preservation/manifest.json`
remain the explicit sources of the growth workbook, historical figures, and
processed plume metrics and normalization coordinates. They are separately
version-pinned dependencies in every relevant run.

The current upstream code expects coordinates and metrics under
`data/datasets/ML_plume_dynamics/`. Setup maps the historical YG identifiers to
G1–G5/C-G6 filenames at those paths, copying the original bytes. Raw files come
directly from the supplied project rather than the old placeholder raw-DID list.
Scientific calculations, cells, and historical source-notebook outputs remain
unchanged by this integration.

To deliberately refresh the input snapshot after a new data publication:

```python
from pathlib import Path
from dataerai_preservation import connect_client
from dataerai_project import pin_project_inputs
with connect_client() as client:
    pin_project_inputs(Path.cwd(), client)
```

Review and commit the new catalogue. If a corrected upload is a separate asset
with the same filename, pass `asset_overrides={'Plumes/G2.h5': 'verified-asset-uuid'}`
to `pin_project_inputs`. Ambiguous filenames stop refresh unless explicitly
resolved. The catalogue retains the superseded asset IDs for audit. Notebook
runs do not silently follow a mutable latest data version.

## Saved provenance

Every run saves notebook source, runtime source code, the input catalogue,
declared environment, installed distribution versions, signed-in user, and
exact input content versions. Each `%%dataerai` cell saves source, sequence,
timing, status, stdout/stderr, rich displays, assigned arrays/tables, and new or
changed files. Saved figures, video, and data retain their original bytes.

The graph links the run to its notebook and inputs, cells in execution order,
outputs to their generating cells, and the executed notebook and terminal
summary to the run. Run `dataerai.finish()` in the final cell. A scientific or
capture error in a managed cell also attempts finalization before propagating
the error. Local spool files remain under `.dataerai/runs/` if a service failure
prevents upload. Closing or forcibly killing the kernel cannot guarantee a
remote terminal summary.

Executed notebook artifacts clear outputs from cells that did not execute in
that run. They include current captured output and cell asset IDs. Historical
outputs are not presented as results of a new execution.

## Validate and republish

```sh
python scripts/validate_dataerai_notebooks.py
python -m pytest tests -q
python scripts/publish_dataerai_notebooks.py --publish
```

The source validator compares all 13 original notebooks against upstream
`84a1761`, including 209 cells and 236 saved outputs. It ignores the managed
integration cells. The new readiness and raw-data notebooks are validated by
execution. Regression tests cover credential/permission failures, paths,
immutable downloads, corruption, filename mapping, closed HDF5 handles,
executed-notebook output isolation, and existing provenance/preservation behavior.

Publication uploads the environment to the project root and standalone base
notebooks to **Base notebooks**, linked to their exact source bundle and
environment. It resumes previous assets rather than duplicating successful
uploads. `preservation/notebook-publication.json` records publication IDs and
server environment resolution. `preservation/project-validation.json` records
live evidence and the extent of scientific execution actually performed.


## Published runtime and local storage

The published bootstrap activates the versioned runtime helpers for inline
Matplotlib output, SDK transfer-log draining, compatibility with the pinned
XRD/layout APIs, and disk-backed large plume arrays. These change execution
support, not the original scientific cells. The declared environment includes
`imageio-ffmpeg==0.6.0`; notebook 5 resolves its packaged binary automatically.

For batch validation, an optional `SRRUO3_SHARED_CACHE` points to a checksum-
addressed cache. An empty cache is populated only from pinned Dataerai assets.
Each run keeps independent working inputs. After its remote completion,
executed-notebook checksum and provenance are verified, derived local raw
working copies can be removed while keeping receipts and outputs. Keep one
verified cache during execution instead of retaining repeated 32 GB workspaces.

Deleting a local download or derived working copy does not delete its server
asset. Before removing all local project material, confirm that raw inputs,
output content, source bundles, environments and successful run receipts are
available remotely, and that intended source changes are committed and pushed
to GitHub. A server content checksum verifies bytes, not HDF5 readability;
validate the required datasets as well. These checks establish reproducibility,
not an independent backup policy for the remote repository.

## Verified complete campaign (2026-10-23 UTC)

All nine published notebooks (0, 00, 1–7) passed without scientific-cell
exclusions, starting with an empty download cache and using only pinned
DataERAI inputs, source and notebook assets. Computation took place in Ubuntu
WSL; DataERAI supplies and preserves the assets, resolved environment definition
and provenance records. This does not assert that computation was hosted by
the DataERAI server.

The immutable aggregate validation asset is
`b8459e89-f908-4393-af21-3802bd88348e`, content
`34352dea-9863-42c1-aa0b-5b133c619721`. The reusable campaign index asset is
`208f2529-1d95-4840-ac09-0bf2200a8164`. Both were downloaded again and verified.
See `preservation/campaign-20261023/` for the nine exact run IDs, base/source/
environment versions, corrected G2/G4 content pins and final publication receipt.

Final audit confirmed all 419 output assets are available and their repository
checksums match local bytes; all six MP4s decode completely; six new metrics
CSVs contain data; 86 NPZ archives retain typed array shapes; twenty exported
figure layouts were visually reviewed. Array-header and rendering checks do
not constitute scientific peer review. The original 209 cells, including 65
code cells and 236 stored historical outputs, remain unchanged against the
pinned upstream baseline. Earlier partial/failed receipts remain history.

To reuse results, download the saved executed notebook, CSV/NPZ assets or
figure/video assets identified by the index. To rerun, use the published base
notebook and its resolved environment; keep exact content pins. After remote
verification and GitHub synchronization, rebuildable local caches, environments
and isolated workspaces are optional. Retain an independent original-recording
backup away from C: if the server's independent backup policy is not established;
GitHub contains code and manifests, not a second copy of the raw recordings.
