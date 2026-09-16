import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest
from dataerai_preservation import (RAW_NAMES, build_manifest, download_archive, extract_verified,
    read_manifest, register_recording_dids, required_bundles, restore, sha256, start_analysis, write_json)
from test_dataerai_notebook import FakeClient, FakeShell, USER_ID


def make_archive(tmp_path, records):
    archive = tmp_path / 'bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data in records.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


def file_record(name, data, bundle='study-data'):
    return {'path': name, 'sha256': hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data), 'bundle': bundle}


class StoreClient(FakeClient):
    def __init__(self, files):
        super().__init__()
        self.files = files
        self.downloads = []

    def download(self, asset_id, dest_dir, **kwargs):
        self.downloads.append((asset_id, kwargs))
        source = self.files[asset_id]
        path = Path(dest_dir) / source.name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        return SimpleNamespace(asset_id=asset_id, content_id=kwargs.get('content_id', 'version-1'),
                               files=[SimpleNamespace(filename=source.name, local_path=str(path))])

    def find_assets(self, did):
        return [SimpleNamespace(asset_id=did)] if did in self.files else []


def setup_repo(tmp_path):
    root = tmp_path / 'repo'
    (root / '.git').mkdir(parents=True)
    (root / 'notebooks').mkdir()
    notebook = root / 'notebooks/1_Abstract.ipynb'
    notebook.write_text(json.dumps({'cells': [], 'metadata': {}}))
    content = {'data/Growth_Parameters.xlsx': b'original workbook'}
    archive = make_archive(tmp_path, content)
    bundle = {'asset_id': 'archive', 'content_id': 'pinned-version', 'filename': archive.name,
              'sha256': sha256(archive), 'size_bytes': archive.stat().st_size}
    manifest = {'files': [file_record(k, v) for k, v in content.items()],
                'bundles': {'study-data': bundle, 'figures': {**bundle}, 'code-and-notebooks': {**bundle}},
                'missing_required_files': ['data/Plumes/plume_recordings/' + n for n in RAW_NAMES]}
    # Figures has its own empty archive.
    folder = tmp_path / 'empty'
    folder.mkdir()
    empty = make_archive(folder, {})
    manifest['bundles']['figures'].update(asset_id='empty', sha256=sha256(empty))
    write_json(root / 'preservation/manifest.json', manifest)
    return root, notebook, StoreClient({'archive': archive, 'empty': empty})


def test_extract_preserves_duplicate_basenames_and_bytes(tmp_path):
    data = {'data/G1/s1.dat': b'first', 'data/G2/s1.dat': b'second'}
    archive = make_archive(tmp_path, data)
    extract_verified(archive, tmp_path / 'restored', [file_record(k, v) for k, v in data.items()])
    assert (tmp_path / 'restored/data/G1/s1.dat').read_bytes() == b'first'
    assert (tmp_path / 'restored/data/G2/s1.dat').read_bytes() == b'second'


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'data/../../escape', 'data\\escape'])
def test_archive_rejects_traversal(tmp_path, name):
    archive = make_archive(tmp_path, {name: b'bad'})
    with pytest.raises(ValueError, match='Unsafe'):
        extract_verified(archive, tmp_path / 'out', [file_record(name, b'bad')])


def test_archive_rejects_links_and_checksum_mismatch(tmp_path):
    archive = tmp_path / 'link.tar'
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo('data/link')
        member.type = tarfile.SYMTYPE
        member.linkname = '/tmp/outside'
        tar.addfile(member)
    with pytest.raises(ValueError, match='Invalid archive'):
        extract_verified(archive, tmp_path / 'out', [file_record('data/link', b'')])
    archive = make_archive(tmp_path, {'data/a': b'bad'})
    with pytest.raises(ValueError, match='checksum'):
        extract_verified(archive, tmp_path / 'out', [file_record('data/a', b'yes')])
    assert not (tmp_path / 'out/data/a').exists()


def test_restore_pins_content_and_rejects_corrupt_download(tmp_path):
    root, _, client = setup_repo(tmp_path)
    restore(root, client, {'study-data'}, tmp_path / 'out')
    assert client.downloads == [('archive', {'content_id': 'pinned-version'})]
    assert (tmp_path / 'out/data/Growth_Parameters.xlsx').read_bytes() == b'original workbook'
    bundle = read_manifest(root)['bundles']['study-data']
    bundle['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA-256'):
        download_archive(client, bundle, tmp_path / 'corrupt')


def test_cache_is_rehashed_and_local_data_never_substituted(tmp_path):
    root, _, client = setup_repo(tmp_path)
    restore(root, client, {'study-data'}, tmp_path / 'out')
    cache = next((root / '.dataerai/downloads').glob('*/*.gz'))
    cache.write_bytes(b'corrupted')
    restore(root, client, {'study-data'}, tmp_path / 'another')
    assert len(client.downloads) == 2


def test_raw_inputs_fail_before_any_download(tmp_path):
    root, _, client = setup_repo(tmp_path)
    with pytest.raises(FileNotFoundError, match='plume-recordings'):
        restore(root, client, required_bundles('4_Plume_Metrics_Extraction.ipynb'), tmp_path / 'out')
    assert not client.downloads
    assert 'plume-recordings' not in required_bundles('6_Target_Temporal_Variation.ipynb')


def test_did_registration_partial_then_complete(tmp_path):
    root, _, client = setup_repo(tmp_path)
    did_map = {}
    for name in RAW_NAMES:
        path = tmp_path / name
        path.write_bytes(b'raw:' + name.encode())
        did = 'did:dataerai:beta:asset:' + name
        client.files[did] = path
        did_map[name] = did
    manifest = register_recording_dids(root, client, dict(list(did_map.items())[:1]))
    assert len(manifest['missing_required_files']) == 5
    with pytest.raises(FileNotFoundError, match='All six'):
        restore(root, client, {'plume-recordings'}, tmp_path / 'out')
    manifest = register_recording_dids(root, client, did_map)
    assert manifest['completeness'] == 'complete'
    restore(root, client, {'plume-recordings'}, tmp_path / 'out')
    for name in RAW_NAMES:
        assert (tmp_path / 'out/data/Plumes/plume_recordings' / name).read_bytes() == b'raw:' + name.encode()
    assert all(kwargs['content_id'] == 'version-1' for _, kwargs in client.downloads[-6:])


def test_wrong_did_file_is_rejected_without_catalogue_mutation(tmp_path):
    root, _, client = setup_repo(tmp_path)
    before = (root / 'preservation/manifest.json').read_bytes()
    client.files['did:dataerai:wrong'] = client.files['archive']
    with pytest.raises(ValueError, match='must contain exactly'):
        register_recording_dids(root, client, {RAW_NAMES[0]: 'did:dataerai:wrong'})
    assert (root / 'preservation/manifest.json').read_bytes() == before


def test_run_saved_files_and_execution_failure_are_preserved(tmp_path):
    import os
    root, notebook, client = setup_repo(tmp_path)
    cwd = Path.cwd()
    run = start_analysis(notebook, client=client, shell=FakeShell(), owner_type='user', owner_id=USER_ID)
    try:
        figure = run.workspace / 'figures/plot.svg'
        figure.parent.mkdir(exist_ok=True)
        figure.write_text('<svg>original vector</svg>')
        run.record_cell(source='raise ValueError("test")', success=False, error=ValueError('test'))
        saved = [u for u in client.uploads if u['kwargs']['metadata']['record_kind'] == 'saved_file']
        assert len(saved) == 1
        assert saved[0]['path'].read_bytes() == figure.read_bytes()
        assert saved[0]['kwargs']['record_type'] == 'figure'
        assert any(r[1] == 'archive' and r[2]['relationship_type'] == 'uses_dependency' for r in client.relationships)
        run.finish()
        assert client.metadata_updates[-1][1]['metadata']['completion']['status'] == 'completed_with_execution_errors'
        assert Path.cwd() == cwd
    finally:
        os.chdir(cwd)


def test_missing_input_restores_working_directory(tmp_path):
    root, _, client = setup_repo(tmp_path)
    notebook = root / 'notebooks/4_Plume_Metrics_Extraction.ipynb'
    notebook.write_text('{}')
    cwd = Path.cwd()
    with pytest.raises(FileNotFoundError):
        start_analysis(notebook, client=client, shell=FakeShell())
    assert Path.cwd() == cwd


def test_saved_file_transient_failure_retries_before_success(tmp_path, monkeypatch):
    import os
    import dataerai_notebook
    monkeypatch.setattr(dataerai_notebook.time, 'sleep', lambda _: None)
    root, notebook, client = setup_repo(tmp_path)
    original_upload = client.upload
    attempts = []

    def flaky_upload(path, **kwargs):
        if str(path).endswith('.svg'):
            attempts.append(str(path))
            if len(attempts) == 1:
                raise RuntimeError('POST complete: HTTP 504')
        return original_upload(path, **kwargs)

    client.upload = flaky_upload
    cwd = Path.cwd()
    run = start_analysis(notebook, client=client, shell=FakeShell(), owner_type='user', owner_id=USER_ID)
    try:
        figure = run.workspace / 'figures/retry.svg'
        figure.parent.mkdir(exist_ok=True)
        figure.write_text('<svg>preserve exactly</svg>')
        run.record_cell(source='save_figure()')
        assert len(attempts) == 2
        saved = [u for u in client.uploads if u['kwargs']['metadata']['record_kind'] == 'saved_file']
        assert len(saved) == 1
        assert saved[0]['path'].read_bytes() == figure.read_bytes()
        assert saved[0]['kwargs']['record_type'] == 'figure'
        run.finish()
    finally:
        os.chdir(cwd)
