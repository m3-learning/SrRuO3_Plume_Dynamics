#!/usr/bin/env python3
"""Publish a standalone, pinned notebook workspace into the raw-data project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataerai_preservation import connect_client, link, sha256, write_json
from dataerai_project import PROJECT_ID, ROOT_COLLECTION_ID, ensure_project_collection, preflight


def bootstrap(name, bundle):
    """The standalone notebook needs only the published environment and SDK."""
    return textwrap.dedent('''\
        from pathlib import Path
        import hashlib, os, shutil, sys, tarfile, tempfile, uuid
        from dataerai import DataeraiClient
        from dataerai.rest import RestClient, load_credentials, resolve_server_url

        # Validate saved login and server before downloading executable sources.
        credentials = load_credentials()
        if (not credentials or not credentials.server_url
                or credentials.server_url.rstrip('/') != 'https://beta.dataerai.com'
                or resolve_server_url().rstrip('/') != 'https://beta.dataerai.com'):
            raise RuntimeError('Sign in first: dataerai auth login --server https://beta.dataerai.com --device')
        del credentials
        with RestClient() as api:
            identity = api.get_json('GET', '/api/me/')
        print('Signed in:', identity['email'])
        binary = str(Path(sys.executable).with_name('dataerai'))
        if not Path(binary).is_file():
            binary = shutil.which('dataerai')
        if not binary:
            raise RuntimeError('Build the collection notebook environment before running this notebook.')
        workspace_parent = Path.cwd() / '.dataerai/workspaces'
        workspace_parent.mkdir(parents=True, exist_ok=True)
        REPO_ROOT = Path(tempfile.mkdtemp(prefix='srruo3-', dir=workspace_parent)).resolve()
        with DataeraiClient(binary_path=binary, socket_path=f'/tmp/srruo3-bootstrap-{uuid.uuid4().hex}.sock') as client:
            client.list_projects()
            if client.auth_status().user_id != identity['id']:
                raise RuntimeError('CLI and SDK identities differ; sign in again before running.')
            result = client.download(BUNDLE_ASSET, str(REPO_ROOT), content_id=BUNDLE_CONTENT)
        if result.content_id != BUNDLE_CONTENT or len(result.files) != 1:
            raise RuntimeError('Source bundle did not match the pinned content version.')
        archive = Path(result.files[0].local_path).resolve()
        if not archive.is_relative_to(REPO_ROOT):
            raise RuntimeError('Source bundle download escaped its workspace.')
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != BUNDLE_SHA256:
            raise RuntimeError('Source bundle checksum mismatch; no source code was loaded.')
        with tarfile.open(archive, 'r:gz') as packed:
            for member in packed.getmembers():
                target = (REPO_ROOT / member.name).resolve()
                if (not member.isfile() or not target.is_relative_to(REPO_ROOT)
                        or Path(member.name).is_absolute() or '\\\\' in member.name
                        or '..' in Path(member.name).parts or target == archive):
                    raise RuntimeError('Unsafe source bundle member.')
            packed.extractall(REPO_ROOT, filter='data')
        # Capture the actual base notebook selected in Jupyter, including this bootstrap.
        candidates = [Path.cwd() / NOTEBOOK_NAME, Path.cwd() / 'notebooks' / NOTEBOOK_NAME]
        current_notebook = next((path for path in candidates if path.is_file()), None)
        if current_notebook is None:
            raise RuntimeError('Start Jupyter in the synced notebook collection directory.')
        shutil.copyfile(current_notebook, REPO_ROOT / 'notebooks' / NOTEBOOK_NAME)
        sys.path.insert(0, str(REPO_ROOT))
        os.chdir(REPO_ROOT)
        print('Verified notebook source workspace:', REPO_ROOT)
        ''').replace('BUNDLE_ASSET', repr(bundle['asset_id'])).replace(
            'BUNDLE_CONTENT', repr(bundle['content_id'])).replace(
            'BUNDLE_SHA256', repr(bundle['sha256'])).replace('NOTEBOOK_NAME', repr(name))


def source_archive():
    paths = set()
    for pattern in ('src/**/*.py', 'dataerai_*.py', 'notebooks/*.ipynb',
                    'notebook-environment/*', 'requirements*.txt', 'pyproject.toml',
                    'preservation/project-inputs.json', 'preservation/manifest.json',
                    'preservation/study.json', 'preservation/key_results.json', 'DATAERAI.md'):
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    archive = ROOT / '.dataerai/publication/workspace-code.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Fix archive metadata so retries reuse exactly the same immutable bytes.
    import gzip
    with archive.open('wb') as sink, gzip.GzipFile(fileobj=sink, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as packed:
            for path in sorted(paths):
                relative = path.relative_to(ROOT).as_posix()
                member = packed.gettarinfo(str(path), arcname=relative)
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ''
                with path.open('rb') as source:
                    packed.addfile(member, source)
    return archive


def publish():
    import nbformat
    from dataerai.rest import RestClient
    from dataerai_notebook import _retry_transient
    receipt_path = ROOT / 'preservation/notebook-publication.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'assets': {}}
    with connect_client() as client:
        report = preflight(client)
        base_collection = ensure_project_collection(client, 'Base notebooks')
        run_collection = ensure_project_collection(client, 'Notebook runs')
        preflight(client, collection_id=base_collection)
        receipt.update(project_id=PROJECT_ID, collection_id=base_collection,
                       run_collection_id=run_collection, server=report['server'])

        def upload(key, path, collection, kind):
            checksum = sha256(path)
            old = receipt['assets'].get(key)
            if old and old['sha256'] == checksum:
                return old
            print(f'Publishing {key}', flush=True)
            if not old:
                # A failed upload may have declared an asset before its bytes
                # arrived. Resume that same workspace entry rather than creating
                # another record on each publication retry.
                candidates = []
                for asset in client.find_collection_assets(collection, query=path.name):
                    if asset.title == path.name:
                        metadata = client.get_metadata(asset.asset_id)
                        if (metadata.metadata or {}).get('path') == key and 'base-notebook-workspace' in (metadata.tags or []):
                            candidates.append(asset.asset_id)
                if len(candidates) > 1:
                    raise ValueError(f'Ambiguous previous publication for {key}')
                if candidates:
                    old = {'asset_id': candidates[0]}

            def transfer():
                # Keep large publication batches independent of daemon lifetime.
                with connect_client() as uploader:
                    if old:
                        return uploader.update_content(old['asset_id'], str(path))
                    return uploader.upload(str(path), title=path.name,
                        owner_type='project', owner_id=PROJECT_ID, collection_id=collection,
                        record_type=kind, tags=['srruo3-plume-dynamics', 'base-notebook-workspace'],
                        metadata={'repository': 'https://github.com/m3-learning/SrRuO3_Plume_Dynamics',
                                  'path': key, 'sha256': checksum})
            result = _retry_transient(transfer)
            record = {'asset_id': result.asset_id, 'content_id': result.content_id,
                      'sha256': checksum, 'filename': path.name}
            receipt['assets'][key] = record
            write_json(receipt_path, receipt)
            return record

        environment = upload('environment', ROOT / 'notebook-environment/pyproject.toml',
                             ROOT_COLLECTION_ID, 'software_code')
        import tomllib
        python_constraint = tomllib.loads((ROOT / 'notebook-environment/pyproject.toml').read_text())['project']['requires-python']
        client.set_metadata(environment['asset_id'], metadata={
            'path': 'notebook-environment/pyproject.toml', 'sha256': environment['sha256'],
            'python_constraint': python_constraint})
        source = upload('source_bundle', source_archive(), base_collection, 'software_code')
        folder = ROOT / '.dataerai/base-notebooks'
        folder.mkdir(parents=True, exist_ok=True)
        for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
            notebook = nbformat.read(path, as_version=4)
            notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap(path.name, source),
                metadata={'tags': ['dataerai-managed-bootstrap']}))
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    cell.execution_count, cell.outputs = None, []
            notebook.metadata['kernelspec'] = {'display_name': 'Python 3 (SrRuO3 Dataerai)',
                                               'language': 'python', 'name': 'python3'}
            notebook.metadata['dataerai'] = {'project_id': PROJECT_ID, 'source_bundle': source,
                                              'environment': environment}
            target = folder / path.name
            nbformat.write(notebook, target)
            published = upload('notebooks/' + path.name, target, base_collection, 'jupyter_notebook')
            for dependency, relation in [(source, 'uses_dependency'), (environment, 'uses_environment')]:
                _retry_transient(lambda: link(client, published['asset_id'], dependency['asset_id'],
                    relation, qualifiers={'content_id': dependency['content_id'], 'sha256': dependency['sha256']}))
        with RestClient() as api:
            response = api.request('GET', f'/api/collections/{base_collection}/environment/')
            receipt['environment_resolution'] = {'status_code': response.status_code,
                                                  'resolution': response.json()}
        write_json(receipt_path, receipt)
        resolution = receipt['environment_resolution']['resolution']
        if (response.status_code != 200 or resolution.get('outcome') != 'resolved'
                or resolution.get('hash_basis') != 'strong'
                or resolution.get('python_constraint') != python_constraint
                or not any(layer.get('asset', {}).get('id') == environment['asset_id']
                           and layer.get('content_id') == environment['content_id']
                           for layer in resolution.get('layers', []))):
            raise RuntimeError('The uploaded notebook environment has not resolved to its pinned Python/spec version.')
    print(json.dumps({'base_collection': base_collection, 'run_collection': run_collection,
                      'assets': len(receipt['assets']), 'environment_status': response.status_code}, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', required=True,
                        help='Upload the environment, versioned sources and clean base notebooks.')
    parser.parse_args()
    publish()
