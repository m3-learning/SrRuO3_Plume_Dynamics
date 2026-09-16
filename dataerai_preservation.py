"""Preserve and restore the SrRuO3 research record through Dataerai.

Archive paths are relative to the research root. Analysis runs work on private
copies so original analysis code can keep using ../data and ../figures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import uuid

REPOSITORY = 'https://github.com/m3-learning/SrRuO3_Plume_Dynamics'
SCHEMA = 'org.dataerai.srruo3-preservation/v1'
RAW_NAMES = [f'{s}_YichenGuo_{d}.h5' for s, d in (
    ('YG063', '08042024'), ('YG065', '09102024'), ('YG066', '09112024'),
    ('YG067', '09122024'), ('YG068', '09132024'), ('YG069', '09152024'))]
SAMPLES = dict(zip(['YG065', 'YG066', 'YG067', 'YG068', 'YG069', 'YG063'],
                   ['G1', 'G2', 'G3', 'G4', 'G5', 'C-G6']))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def read_manifest(root):
    return json.loads((Path(root) / 'preservation/manifest.json').read_text())


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative:
        raise ValueError(f'Unsafe preservation path: {relative!r}')
    root = Path(root).resolve()
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes destination: {relative}')
    return result


def record_type(path):
    suffix = Path(path).suffix.lower()
    if suffix == '.ipynb':
        return 'jupyter_notebook'
    if suffix in {'.png', '.svg', '.tif', '.tiff', '.jpg', '.jpeg', '.pdf'}:
        return 'figure'
    if suffix in {'.py', '.toml', '.yml', '.yaml'}:
        return 'software_code'
    return 'analysis'


def git_files(root):
    return subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')[:-1]


def inventory(root, data_root, raw_root=None):
    """Inventory explicit study directories, including ignored HDF5 recordings."""
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    sources = {}
    for relative in git_files(root):
        path = root / relative
        if path.is_file() and not path.is_symlink():
            sources[relative] = path
    for path in sorted(data_root.rglob('*')):
        if path.is_file() and not path.is_symlink() and not any(x.startswith('.') for x in path.relative_to(data_root).parts):
            sources['data/' + path.relative_to(data_root).as_posix()] = path
    if raw_root:
        for name in RAW_NAMES:
            path = Path(raw_root) / name
            if path.is_file() and not path.is_symlink():
                sources['data/Plumes/plume_recordings/' + name] = path
    # Integration sources may not have been committed yet. Include only an explicit allowlist.
    for pattern in ('dataerai_*.py', 'requirements-dataerai.txt', 'DATAERAI.md',
                    'scripts/*dataerai*.py', 'preservation/key_results.json',
                    'preservation/study.json', 'notebooks/0_Dataerai_Preservation.ipynb'):
        for path in root.glob(pattern):
            if path.is_file():
                sources[path.relative_to(root).as_posix()] = path
    sources.pop('preservation/manifest.json', None)
    sources.pop('preservation/validation.json', None)
    records = []
    for relative, path in sorted(sources.items()):
        if relative.startswith('data/Plumes/plume_recordings/'):
            bundle = 'plume-recordings'
        elif relative.startswith('data/'):
            bundle = {'AFM': 'afm', 'XRD_RSM': 'xrd', 'TargetMicroscopy': 'target-microscopy', 'Plumes': 'plume-metrics'}.get(relative.split('/')[1], 'study-data')
        elif relative.startswith('figures/'):
            bundle = 'figures'
        else:
            bundle = 'code-and-notebooks'
        records.append({'path': relative, 'bundle': bundle, 'sha256': sha256(path),
                        'size_bytes': path.stat().st_size,
                        'sample_ids': [sample for sample in SAMPLES if sample in relative]})
    missing = ['data/Plumes/plume_recordings/' + name for name in RAW_NAMES
               if 'data/Plumes/plume_recordings/' + name not in sources]
    return sources, records, missing


def build_manifest(root, data_root, raw_root=None):
    root = Path(root).resolve()
    sources, records, missing = inventory(root, data_root, raw_root)
    archive_root = root / '.dataerai/archives'
    archive_root.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for name in sorted({r['bundle'] for r in records}):
        members = [r for r in records if r['bundle'] == name]
        identity = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
        archive = archive_root / f'{name}-{identity[:16]}.tar.gz'
        if not archive.exists():
            temporary = archive.with_suffix('.tmp')
            with tarfile.open(temporary, 'w:gz', compresslevel=1) as tar:
                for item in members:
                    path = sources[item['path']]
                    tar.add(path, arcname=item['path'], recursive=False)
                    if sha256(path) != item['sha256']:
                        raise RuntimeError(f'File changed while packing: {path}')
            temporary.replace(archive)
        bundles[name] = {'filename': archive.name, 'sha256': sha256(archive),
                         'size_bytes': archive.stat().st_size, 'file_count': len(members)}
    previous_path = root / 'preservation/manifest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    # Resume previously published immutable bundles; never carry IDs across changed bytes.
    for name, bundle in bundles.items():
        old = previous.get('bundles', {}).get(name, {})
        if old.get('sha256') == bundle['sha256']:
            bundle.update({key: old[key] for key in ('asset_id', 'content_id', 'verified_download') if key in old})
    old_raw = previous.get('bundles', {}).get('plume-recordings', {})
    if old_raw.get('kind') == 'files' and 'plume-recordings' not in bundles:
        bundles['plume-recordings'] = old_raw
        records.extend(r for r in previous['files'] if r['bundle'] == 'plume-recordings')
        missing = [p for p in missing if Path(p).name not in old_raw['recordings']]
    manifest = {'schema': SCHEMA, 'repository': REPOSITORY,
                'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                'source_snapshot': 'Working-tree files identified by their SHA-256; source_commit is the Git baseline',
                'data_source': 'Explicit source folder supplied to preservation notebook; see per-file hashes',
                'samples': SAMPLES, 'missing_required_files': missing,
                'completeness': 'missing-raw-recordings' if missing else 'complete',
                'files': records, 'bundles': bundles,
                'publication': previous.get('publication', {})}
    write_json(previous_path, manifest)
    return manifest


def connect_client():
    from dataerai import DataeraiClient
    binary = os.getenv('DATAERAI_BINARY') or shutil.which('dataerai')
    if not binary:
        raise RuntimeError('Install requirements-dataerai.txt and run dataerai auth login.')
    client = DataeraiClient(binary_path=binary, socket_path=f'/tmp/srruo3-{uuid.uuid4().hex}.sock')
    return client


def publish(root, client, destination='SrRuO3 Plume Dynamics / Preservation', *, verify=True):
    """Upload immutable bundles and verify downloaded bytes before recording success."""
    root = Path(root).resolve()
    manifest = read_manifest(root)
    resolved = client.ensure_collection_path(destination, create_project=True)
    previous_destination = manifest.get('publication', {}).get('collection_id')
    if previous_destination and previous_destination != resolved.collection_id:
        raise ValueError('This catalogue is already pinned to a different Dataerai collection.')
    manifest['publication'] = {'project_id': resolved.project_id, 'collection_id': resolved.collection_id,
                               'collection_path': destination}
    write_json(root / 'preservation/manifest.json', manifest)
    for name, bundle in manifest['bundles'].items():
        if bundle.get('kind') == 'files':
            continue
        archive = root / '.dataerai/archives' / bundle['filename']
        if not bundle.get('asset_id'):
            if not archive.is_file() or sha256(archive) != bundle['sha256']:
                raise ValueError(f'Archive missing or checksum mismatch: {name}. Build from the source data first.')
            print(f'Preserving {name}: {bundle["file_count"]} files, {bundle["size_bytes"]:,} bytes', flush=True)
            result = client.upload(str(archive), title=f'SrRuO3 {name} sha256:{bundle["sha256"]}',
                                   owner_type='project', owner_id=resolved.project_id,
                                   collection_id=resolved.collection_id, record_type='dataset',
                                   tags=['srruo3-plume-dynamics', 'preservation', name],
                                   metadata={'schema': SCHEMA, 'repository': REPOSITORY,
                                             'source_commit': manifest['source_commit'],
                                             'bundle': name, 'sha256': bundle['sha256'],
                                             'files': [r for r in manifest['files'] if r['bundle'] == name],
                                             'completeness': manifest['completeness']})
            bundle.update(asset_id=result.asset_id, content_id=result.content_id)
            write_json(root / 'preservation/manifest.json', manifest)
        if verify and not bundle.get('verified_download'):
            with tempfile.TemporaryDirectory(prefix='srruo3-verify-') as folder:
                download_archive(client, bundle, Path(folder))
            bundle['verified_download'] = True
            write_json(root / 'preservation/manifest.json', manifest)
    # Publish human-readable scientific metadata and key results as searchable records.
    for relative, kind in [('preservation/study.json', 'protocol_workflow'),
                           ('preservation/key_results.json', 'analysis')]:
        path = root / relative
        result = client.upload(str(path), title=f'SrRuO3 {path.stem} sha256:{sha256(path)}',
                               owner_type='project', owner_id=resolved.project_id,
                               collection_id=resolved.collection_id, record_type=kind,
                               tags=['srruo3-plume-dynamics', path.stem],
                               metadata=json.loads(path.read_text()))
        manifest['publication'][path.stem + '_asset_id'] = result.asset_id
        link(client, result.asset_id, manifest['bundles']['code-and-notebooks']['asset_id'], 'derived_from')
        write_json(root / 'preservation/manifest.json', manifest)
    # The catalogue itself is discoverable in Dataerai, with lineage to every bundle.
    catalog = root / '.dataerai/catalog.json'
    write_json(catalog, manifest)
    result = client.upload(str(catalog), title=f'SrRuO3 preservation catalogue sha256:{sha256(catalog)}',
                           owner_type='project', owner_id=resolved.project_id, collection_id=resolved.collection_id,
                           record_type='protocol_workflow', tags=['srruo3-plume-dynamics', 'preservation-catalogue'],
                           metadata={'schema': SCHEMA, 'completeness': manifest['completeness'],
                                     'missing_required_files': manifest['missing_required_files']})
    for bundle in manifest['bundles'].values():
        entries = bundle['recordings'].values() if bundle.get('kind') == 'files' else [bundle]
        for entry in entries:
            link(client, result.asset_id, entry['asset_id'], 'documents')
    manifest['publication']['catalogue_asset_id'] = result.asset_id
    manifest['publication']['catalogue_content_id'] = result.content_id
    write_json(root / 'preservation/manifest.json', manifest)
    return manifest


def link(client, source, target, relationship_type, **kwargs):
    try:
        client.create_relationship(source, target, relationship_type=relationship_type, **kwargs)
    except Exception as error:
        if getattr(error, 'code', None) != 'ERR_RELATIONSHIP_EXISTS':
            raise


def download_archive(client, bundle, destination):
    if not bundle.get('asset_id') or not bundle.get('content_id'):
        raise RuntimeError('This bundle has not been preserved. Run notebook 0 first.')
    result = client.download(bundle['asset_id'], str(destination), content_id=bundle['content_id'])
    if len(result.files) != 1:
        raise ValueError('Expected exactly one archive in the pinned Dataerai content version')
    archive = Path(result.files[0].local_path).resolve()
    if not archive.is_relative_to(Path(destination).resolve()) or sha256(archive) != bundle['sha256']:
        raise ValueError('Dataerai download failed SHA-256 verification')
    return archive


def extract_verified(archive, destination, records):
    """Reject unexpected members, links, traversal, duplicates and corrupt data."""
    destination = Path(destination).resolve()
    expected = {item['path']: item for item in records}
    with tarfile.open(archive, 'r:*') as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Archive membership differs from preservation catalogue')
        for member in members:
            safe_path(destination, member.name)
            if not member.isfile() or member.size != expected[member.name]['size_bytes']:
                raise ValueError(f'Invalid archive member: {member.name}')
        for member in members:
            target = safe_path(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as sink:
                shutil.copyfileobj(source, sink)
            if sha256(target) != expected[member.name]['sha256']:
                target.unlink()
                raise ValueError(f'File checksum mismatch: {member.name}')


def required_bundles(notebook):
    name = Path(notebook).name
    common = {'study-data', 'figures'}
    if name.startswith('2_'):
        return common | {'target-microscopy'}
    if name.startswith('3_'):
        return common | {'afm', 'xrd'}
    if name.startswith(('4_', '5_')):
        return common | {'plume-recordings', 'plume-metrics'}
    if name.startswith(('6_', '7_')):
        return common | {'plume-metrics'}
    return common


def restore(root, client, bundles, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    manifest = read_manifest(root)
    missing = set(bundles) - manifest['bundles'].keys()
    if missing:
        raise FileNotFoundError('Missing preserved input bundles: ' + ', '.join(sorted(missing)) +
                                '. Supply the six HDF5 recordings to notebook 0. No data are synthesized or skipped.')
    # Check the complete requirement set before downloading or running any analysis.
    for name in bundles:
        bundle = manifest['bundles'][name]
        if bundle.get('kind') == 'files':
            if set(bundle['recordings']) != set(RAW_NAMES):
                raise FileNotFoundError('All six recording DIDs are required for this notebook.')
            continue
        if not bundle.get('content_id'):
            raise RuntimeError(f'{name} is not published; run notebook 0 first.')
    cache = root / '.dataerai/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    for name in sorted(bundles):
        bundle = manifest['bundles'][name]
        if bundle.get('kind') == 'files':
            for filename, entry in bundle['recordings'].items():
                target = safe_path(destination, 'data/Plumes/plume_recordings/' + filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache, prefix='raw-') as folder:
                    downloaded = download_archive(client, entry, Path(folder))
                    shutil.copyfile(downloaded, target)
            continue
        cache_dir = cache / bundle['sha256']
        cache_dir.mkdir(exist_ok=True)
        cached = cache_dir / bundle['filename']
        if not cached.is_file() or sha256(cached) != bundle['sha256']:
            with tempfile.TemporaryDirectory(dir=cache, prefix='incoming-') as folder:
                downloaded = download_archive(client, bundle, Path(folder))
                shutil.copyfile(downloaded, cached)
        extract_verified(cached, destination, [r for r in manifest['files'] if r['bundle'] == name])
    return manifest


def start_analysis(notebook_path, **kwargs):
    """Restore inputs, then record cells without changing their scientific code."""
    from dataerai_notebook import NotebookProvenance

    class PlumeRun(NotebookProvenance):
        def _capture_dependencies(self):
            # Preserve the actual local code used by this execution, even after edits.
            paths = set(self.repo_root.glob('src/sro_sto_plume/**/*.py'))
            for pattern in ('dataerai_*.py', 'requirements*.txt', 'pyproject.toml'):
                paths.update(self.repo_root.glob(pattern))
            archive = self.run_dir / 'runtime-code.tar.gz'
            records = []
            with tarfile.open(archive, 'w:gz') as tar:
                for path in sorted(paths):
                    if path.is_file() and not path.is_symlink():
                        relative = path.relative_to(self.repo_root).as_posix()
                        digest = sha256(path)
                        tar.add(path, arcname=relative, recursive=False)
                        if sha256(path) != digest:
                            raise RuntimeError(f'Code changed while capturing: {relative}')
                        records.append({'path': relative, 'sha256': digest})
            asset = self._upload(archive,
                title=f'SrRuO3 runtime code {self.run_id}',
                description='Exact local Python sources and declared dependencies for this run.',
                metadata={**self._base_metadata('notebook_dependency'),
                          'dependency': {'role': 'software_source', 'sha256': sha256(archive), 'files': records}},
                tags=self._tags('runtime_code'))
            self._link(self.run_asset_id, asset, 'uses_dependency')

        def record_cell(self, **cell):
            asset_id = super().record_cell(**cell)
            self.capture_files(asset_id)
            return asset_id

        def file_state(self):
            return {p.relative_to(self.workspace).as_posix(): sha256(p)
                    for folder in ('data', 'figures') for p in (self.workspace / folder).rglob('*')
                    if p.is_file() and not p.is_symlink()}

        def _record_type_for_upload(self, path, metadata):
            if metadata.get('record_kind') == 'saved_file':
                return record_type(path)
            return super()._record_type_for_upload(path, metadata)

        def capture_files(self, parent):
            state = self.file_state()
            for relative, digest in state.items():
                if self.file_hashes.get(relative) == digest:
                    continue
                path = self.workspace / relative
                preserved = self.run_dir / 'files' / f'{self.cell_count:04d}' / relative
                preserved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, preserved)
                asset_id = self._upload(preserved,
                    title=f'SrRuO3 {self.run_id} cell {self.cell_count} {relative}'[:255],
                    description='Original bytes of a file saved by this notebook cell.',
                    tags=self._tags('saved_file'), metadata={**self._base_metadata('saved_file'),
                        'path': relative, 'sha256': digest, 'size_bytes': path.stat().st_size})
                self._link(asset_id, parent, 'generated_by')
            self.file_hashes = state

        def finish(self):
            try:
                if self.running:
                    self.capture_files(self.last_cell_asset_id or self.run_asset_id)
                return super().finish()
            finally:
                os.chdir(self.original_cwd)
        close = finish

    notebook_path = Path(notebook_path).resolve()
    run = PlumeRun(notebook_path, **kwargs)
    run.original_cwd = Path.cwd()
    run.workspace = run.run_dir / 'workspace'
    client = run.client or connect_client()
    if run._owns_client:
        client.connect()
    run.client = client
    run._client_connected = True
    if run._owns_client:
        run.binary_path = os.getenv('DATAERAI_BINARY') or shutil.which('dataerai')
    try:
        needs = required_bundles(notebook_path)
        manifest = restore(run.repo_root, client, needs, run.workspace)
        if run.owner_type == 'auto' and not run.owner_id:
            resolved = client.ensure_collection_path('SrRuO3 Plume Dynamics / Notebook runs', create_project=True)
            run.owner_type, run.owner_id, run.collection_id = 'project', resolved.project_id, resolved.collection_id
        run.file_hashes = run.file_state()
        # The notebooks' ../data and ../figures now refer only to downloaded copies.
        working = run.workspace / 'notebooks'
        working.mkdir(parents=True, exist_ok=True)
        import sys
        sys.path.insert(0, str(run.repo_root / 'src'))
        os.chdir(working)
        run.start()
        for name in sorted(needs | {'code-and-notebooks'}):
            bundle = manifest['bundles'][name]
            entries = bundle['recordings'].values() if bundle.get('kind') == 'files' else [bundle]
            for entry in entries:
                run._link(run.run_asset_id, entry['asset_id'], 'uses_dependency',
                     qualifiers={'bundle': name, 'content_id': entry['content_id'], 'sha256': entry['sha256']})
        return run
    except BaseException:
        os.chdir(run.original_cwd)
        if run._owns_client:
            client.close()
        raise


def register_recording_dids(root, client, did_by_filename):
    """Pin later-uploaded raw data by DID, content version, size, and SHA-256.

    Each DID must resolve to exactly one readable asset containing exactly one
    file with the expected original filename. Partial registration is resumable.
    """
    root = Path(root).resolve()
    manifest = read_manifest(root)
    unknown = set(did_by_filename) - set(RAW_NAMES)
    if unknown:
        raise ValueError(f'Unexpected recording names: {sorted(unknown)}')
    bundle = manifest['bundles'].get('plume-recordings', {'kind': 'files', 'recordings': {}})
    if bundle.get('kind') != 'files':
        raise ValueError('Raw recordings are already preserved as an archive.')
    for name, did in did_by_filename.items():
        if not did or not did.startswith('did:dataerai:'):
            raise ValueError(f'Provide the Dataerai DID for {name}')
        matches = client.find_assets(did)
        if len(matches) != 1:
            raise LookupError(f'{did}: expected one accessible asset, found {len(matches)}')
        with tempfile.TemporaryDirectory(prefix='srruo3-raw-') as directory:
            result = client.download(matches[0].asset_id, directory)
            if len(result.files) != 1 or result.files[0].filename != name:
                raise ValueError(f'{did} must contain exactly one file named {name}')
            path = Path(result.files[0].local_path).resolve()
            if not path.is_relative_to(Path(directory).resolve()):
                raise ValueError('Downloaded file escaped the staging directory')
            item = {'did': did, 'asset_id': result.asset_id, 'content_id': result.content_id,
                    'filename': name, 'sha256': sha256(path), 'size_bytes': path.stat().st_size,
                    'verified_download': True}
            bundle['recordings'][name] = item
        relative = 'data/Plumes/plume_recordings/' + name
        manifest['files'] = [r for r in manifest['files'] if r['path'] != relative]
        manifest['files'].append({'path': relative, 'bundle': 'plume-recordings',
                                 'sha256': item['sha256'], 'size_bytes': item['size_bytes'],
                                 'sample_ids': [name.split('_')[0]]})
        manifest['bundles']['plume-recordings'] = bundle
        manifest['missing_required_files'] = ['data/Plumes/plume_recordings/' + n
            for n in RAW_NAMES if n not in bundle['recordings']]
        manifest['completeness'] = 'missing-raw-recordings' if manifest['missing_required_files'] else 'complete'
        write_json(root / 'preservation/manifest.json', manifest)
    return manifest
