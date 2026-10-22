"""Authenticated, version-pinned inputs from the SrRuO3 Data project."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from uuid import UUID

SERVER = 'https://beta.dataerai.com'
PROJECT_ID = '99185d30-fe63-497c-91b4-852a02574d49'
ROOT_COLLECTION_ID = '7461b56c-7dc6-4677-b0c8-6d165bfc1eed'
LOGIN = 'dataerai auth login --server https://beta.dataerai.com --device'
SAMPLE_NAMES = dict(zip(('YG065', 'YG066', 'YG067', 'YG068', 'YG069', 'YG063'),
                        ('G1', 'G2', 'G3', 'G4', 'G5', 'C-G6')))


def check_notebook_environment(root):
    """Fail before downloads when the repository's declared runtime is missing."""
    import importlib.metadata
    import sys
    import tomllib
    from packaging.requirements import Requirement
    spec = tomllib.loads((Path(root) / 'notebook-environment/pyproject.toml').read_text())['project']
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use the SrRuO3 Python 3.11 notebook environment. See DATAERAI.md.')
    missing = []
    for declaration in spec['dependencies']:
        requirement = Requirement(declaration)
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(declaration)
            continue
        if version not in requirement.specifier:
            missing.append(declaration)
        if requirement.url:
            direct = importlib.metadata.distribution(requirement.name).read_text('direct_url.json')
            installed = json.loads(direct) if direct else {}
            origin, _, revision = requirement.url.removeprefix('git+').rpartition('@')
            if (installed.get('url') != origin
                    or installed.get('vcs_info', {}).get('commit_id') != revision):
                missing.append(declaration)
    if missing:
        raise RuntimeError('Notebook environment is incomplete or incompatible: ' + ', '.join(missing))
    # AFM-Learn 0.10 imports the abandoned igor 0.3 reader, which fails with
    # NumPy 1.26. Use igor2's maintained binary reader through that import path.
    import igor
    import igor2.binarywave
    igor.binarywave = igor2.binarywave
    sys.modules['igor.binarywave'] = igor2.binarywave
    return {'python': sys.version.split()[0], 'declared_dependencies_checked': len(spec['dependencies']),
            'igor_binary_reader': importlib.metadata.version('igor2')}


def preflight(client, *, project_id=PROJECT_ID, collection_id=ROOT_COLLECTION_ID,
              require_write=True, rest=None):
    """Check real server identity and permissions before transfers or execution.

    RestClient refreshes saved credentials through the supported SDK. Neither
    credentials nor authorization headers are included in the returned report.
    """
    from dataerai.rest import RestClient, load_credentials, resolve_server_url
    if rest is None:
        credentials = load_credentials()
        if not credentials or not credentials.server_url:
            raise RuntimeError(f'No saved Dataerai login. Run: {LOGIN}')
        if credentials.server_url.rstrip('/') != SERVER or resolve_server_url().rstrip('/') != SERVER:
            raise RuntimeError(f'This study requires {SERVER}. Run: {LOGIN}')
    for value in (project_id, collection_id):
        UUID(value)
    try:
        with nullcontext(rest) if rest is not None else RestClient() as api:
            if api.base_url.rstrip('/') != SERVER:
                raise RuntimeError(f'This study requires {SERVER}. Run: {LOGIN}')
            identity = api.get_json('GET', '/api/me/')
            if not identity.get('id') or not identity.get('email'):
                raise RuntimeError('Dataerai did not confirm the signed-in identity.')
            project = api.get_json('GET', f'/api/projects/{project_id}/')
            source = api.get_json('GET', f'/api/collections/{ROOT_COLLECTION_ID}/')
            destination = api.get_json('GET', f'/api/collections/{collection_id}/')
            if source.get('owner_project_id') != PROJECT_ID or source.get('effective_flags', 0) & 0x015 != 0x015:
                raise PermissionError('Read access to the SrRuO3 raw-data collection is required.')
            if destination.get('owner_project_id') != project_id:
                raise PermissionError('Notebook destination belongs to a different project.')
            required = 0x03F if require_write else 0x015
            if destination.get('effective_flags', 0) & required != required:
                raise PermissionError('Notebook destination lacks the required read/write permissions.')
            # Exercise the transfer daemon too: it uses the CLI credential store.
            client.list_projects()
            auth = client.auth_status()
            if auth.user_id != identity['id']:
                raise RuntimeError('The CLI and REST SDK are signed in as different users. Sign in again.')
            return {'server': SERVER, 'user_id': identity['id'], 'user_email': identity['email'],
                    'project_id': project['id'], 'collection_id': collection_id,
                    'read_checked': True, 'write_checked': require_write}
    except (PermissionError, RuntimeError):
        raise
    except Exception as error:
        # SDK exceptions may contain request details; do not copy them into notebooks.
        raise RuntimeError(f'Dataerai credential/access check failed. Run: {LOGIN}') from None


def _relative(path):
    from dataerai_preservation import safe_path
    # safe_path also rejects absolute paths, traversal, and backslashes.
    safe_path(Path('/tmp/srruo3-path-check'), path)
    return path


def pin_project_inputs(root, client, *, rest=None):
    """Record exact native content versions and repository-verified checksums."""
    from dataerai.rest import RestClient
    from dataerai_preservation import write_json
    preflight(client, require_write=False, rest=rest)
    manifest = client.get_collection_manifest(ROOT_COLLECTION_ID)
    files, paths, unavailable = [], set(), []
    with nullcontext(rest) if rest is not None else RestClient() as api:
        for entry in manifest.files:
            relative = _relative(entry.relative_path)
            # Published notebook workspaces must never become raw-data inputs.
            if relative.split('/')[0] not in {'AFM', 'TargetMicroscopy', 'XRD_RSM', 'Plumes'}:
                continue
            if relative in paths:
                raise ValueError(f'Ambiguous duplicate input path: {relative}')
            paths.add(relative)
            if not entry.is_downloadable or entry.external:
                raise ValueError(f'Input is not a verified native download: {relative}')
            try:
                asset = api.get_json('GET', f'/api/assets/{entry.asset_id}/')
            except Exception as error:
                if getattr(error, 'status_code', None) != 404:
                    raise
                unavailable.append({'path': relative, 'asset_id': entry.asset_id,
                                    'reason': 'Asset detail returned 404; not an executable input.'})
                continue
            content = asset.get('current_content') or {}
            matches = [f for f in content.get('files', []) if f['filename'] == entry.filename]
            if asset.get('owner_id') != PROJECT_ID or content.get('status') != 'available' or len(matches) != 1:
                raise ValueError(f'Input is unavailable or belongs to another project: {relative}')
            item = matches[0]
            checksum = item.get('checksum', '')
            if not re.fullmatch(r'(sha256|blake3):[0-9a-f]{64}', checksum):
                raise ValueError(f'Input has no supported verified checksum: {relative}')
            if not item.get('uploaded') or item['size'] != entry.size_bytes:
                raise ValueError(f'Input changed while pinning: {relative}')
            files.append({'path': relative, 'filename': entry.filename,
                          'asset_id': entry.asset_id, 'did': asset.get('did'),
                          'content_id': content['id'], 'size_bytes': item['size'], 'checksum': checksum})
    recordings = {f['path'] for f in files if f['path'].endswith('.h5')}
    expected = {f'Plumes/{sample}.h5' for sample in SAMPLE_NAMES.values()}
    if not expected <= recordings:
        raise FileNotFoundError(f'Missing raw recordings: {sorted(expected - recordings)}')
    result = {'schema': 'org.dataerai.srruo3-project-inputs/v1', 'server': SERVER,
              'project_id': PROJECT_ID, 'root_collection_id': ROOT_COLLECTION_ID,
              'files': sorted(files, key=lambda f: f['path']), 'unavailable_files': unavailable}
    write_json(Path(root) / 'preservation/project-inputs.json', result)
    return result


def verify_file(path, record):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record['size_bytes']:
        return False
    algorithm, expected = record['checksum'].split(':', 1)
    if algorithm == 'blake3':
        from blake3 import blake3
        digest = blake3()
    elif algorithm == 'sha256':
        digest = hashlib.sha256()
    else:
        raise ValueError(f'Unsupported input checksum: {algorithm}')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == expected


def notebook_inputs(root, notebook, *, input_paths=None):
    catalogue = json.loads((Path(root) / 'preservation/project-inputs.json').read_text())
    if catalogue['server'] != SERVER or catalogue['project_id'] != PROJECT_ID:
        raise ValueError('Project input catalogue points to a different server/project.')
    name = Path(notebook).name
    prefixes = ('TargetMicroscopy/',) if name.startswith('2_') else (
        ('AFM/10um/', 'AFM/5um/', 'XRD_RSM/') if name.startswith('3_') else (
            ('Plumes/',) if name.startswith(('4_', '5_')) else ()))
    files = catalogue['files']
    for item in files:
        _relative(item['path'])
        _relative(item['filename'])
        UUID(item['asset_id'])
        UUID(item['content_id'])
        if not re.fullmatch(r'(sha256|blake3):[0-9a-f]{64}', item['checksum']):
            raise ValueError('Unsupported or missing input checksum.')
    if len({f['path'] for f in files}) != len(files):
        raise ValueError('Duplicate input paths in catalogue.')
    if input_paths is not None:
        wanted = set(input_paths)
        selected = [f for f in files if f['path'] in wanted]
        if {f['path'] for f in selected} != wanted:
            raise FileNotFoundError('Requested input is absent from the pinned project catalogue.')
        return selected
    missing = [f['path'] for f in catalogue.get('unavailable_files', []) if f['path'].startswith(prefixes)]
    if missing:
        raise FileNotFoundError(f'Notebook inputs are inaccessible: {missing}')
    return [f for f in files if f['path'].startswith(prefixes)]


def restore_project_inputs(root, client, records, destination):
    """Download only selected pinned files, validate cache bytes, keep folder paths."""
    from dataerai_preservation import safe_path
    root, destination = Path(root), Path(destination)
    targets = [(item, safe_path(destination, 'data/' + _relative(item['path']))) for item in records]
    for item, target in targets:
        algorithm, digest = item['checksum'].split(':', 1)
        cache = root / '.dataerai/project-downloads' / algorithm / digest
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not verify_file(cache, item):
            with tempfile.TemporaryDirectory(dir=cache.parent) as staging:
                print(f"Downloading pinned input {item['path']} ({item['size_bytes']:,} bytes)", flush=True)
                result = client.download(item['asset_id'], staging, content_id=item['content_id'])
                if result.content_id != item['content_id'] or len(result.files) != 1:
                    raise ValueError('Download did not return the pinned single-file content version.')
                downloaded = Path(result.files[0].local_path).resolve()
                if not downloaded.is_relative_to(Path(staging).resolve()) or result.files[0].filename != item['filename']:
                    raise ValueError('Downloaded input has an unexpected path or filename.')
                if not verify_file(downloaded, item):
                    raise ValueError(f"Input checksum/size mismatch: {item['path']}")
                downloaded.replace(cache)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache, target)
    return records


def map_legacy_inputs(workspace):
    """Map preserved YG identifiers to upstream's current sample names and paths."""
    workspace = Path(workspace)
    for folder, new_folder in [('frame_normalize_dataset', 'frame_normalize_dataset'),
                               ('plume_metrics', 'plume_metrics')]:
        for source in (workspace / 'data/Plumes' / folder).glob('*'):
            stem = source.name
            for old, new in SAMPLE_NAMES.items():
                if stem.startswith(old + '_'):
                    stem = new + stem[len(old):]
                    break
            target = workspace / 'data/datasets/ML_plume_dynamics' / new_folder / stem
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def ensure_project_collection(client, title, *, parent_id=ROOT_COLLECTION_ID):
    matches = [c for c in client.list_collections(owner_type='project', owner_id=PROJECT_ID)
               if c.title == title and c.parent_id == parent_id]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous collection: {title}')
    if matches:
        return matches[0].collection_id
    return client.create_collection(title, owner_type='project', owner_id=PROJECT_ID,
                                    parent_id=parent_id).collection_id
