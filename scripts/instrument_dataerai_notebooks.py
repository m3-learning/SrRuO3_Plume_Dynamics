#!/usr/bin/env python3
"""Add capture around original cells; never clear stored outputs or rewrite science."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREFIX = '%%dataerai\n'


def cell(kind, source, tag):
    result = {'cell_type': kind, 'metadata': {'tags': [tag]},
              'source': source.splitlines(keepends=True),
              'id': hashlib.sha256((tag + source).encode()).hexdigest()[:24]}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def instrument(path):
    notebook = json.loads(path.read_text())
    relative = path.relative_to(ROOT).as_posix()
    originals = []
    for existing in notebook['cells']:
        if any(t.startswith('dataerai-managed-') for t in existing.get('metadata', {}).get('tags', [])):
            continue
        if existing['cell_type'] == 'code':
            source = ''.join(existing['source'])
            if source.startswith(PREFIX):
                source = source[len(PREFIX):]
            if source.strip():
                existing['source'] = (PREFIX + source).splitlines(keepends=True)
        originals.append(existing)
    insertion = next((i for i, c in enumerate(originals) if c['cell_type'] == 'code'), len(originals))
    setup = f'''from pathlib import Path
import os
import sys

# Works when Jupyter starts at the repository root or in a notebook folder.
REPO_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
                 if (p / "dataerai_preservation.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dataerai_preservation import start_analysis

dataerai = start_analysis(
    REPO_ROOT / {relative!r},
    owner_type=os.getenv("DATAERAI_OWNER_TYPE", "auto"),
    owner_id=os.getenv("DATAERAI_OWNER_ID") or None,
    collection_id=os.getenv("DATAERAI_COLLECTION_ID") or None,
)
dataerai
'''
    notes = '''## Dataerai inputs and preservation

Use the Python 3.11 environment in `notebook-environment/pyproject.toml` and sign in with
`dataerai auth login --server https://beta.dataerai.com` (add `--device` for a remote machine).
The next cell checks the login, server, project access and runtime before any
transfer or analysis. It downloads version-pinned inputs from the SrRuO3 Data
project, verifies BLAKE3/SHA-256 hashes, and opens a separate run workspace.
Historical figures and processed inputs come from the preservation catalogue.
Each `%%dataerai` cell records its source,
outputs, figures, tables, and provenance. Run the final cell to upload the run
summary and executed notebook. See `DATAERAI.md` for setup and input catalogues.
'''
    notebook['cells'] = [*originals[:insertion], cell('markdown', notes, 'dataerai-managed-notes'),
        cell('code', setup, 'dataerai-managed-setup'), *originals[insertion:],
        cell('markdown', '## Preserve the completed run\n\nUploads final files and the run summary, then restores the starting directory.\n', 'dataerai-managed-finish-notes'),
        cell('code', 'dataerai.finish()\n', 'dataerai-managed-finish')]
    for i, c in enumerate(notebook['cells']):
        c.setdefault('id', hashlib.sha256((relative + str(i) + ''.join(c['source'])).encode()).hexdigest()[:24])
    notebook['nbformat_minor'] = max(5, notebook.get('nbformat_minor', 0))
    rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    baseline = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        '84a1761bbc15512d61a755f0fc6265d0391f76b4:' + relative])
    if b'\r\n' in baseline:
        rendered = rendered.replace('\n', '\r\n')
    path.write_bytes(rendered.encode())


if __name__ == '__main__':
    paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '*.ipynb'], text=True).splitlines()
    paths = [p for p in paths if p != 'notebooks/0_Dataerai_Preservation.ipynb']
    for relative in paths:
        instrument(ROOT / relative)
    print(f'Instrumented {len(paths)} original notebooks; existing outputs retained.')
