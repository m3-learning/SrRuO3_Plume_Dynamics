import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataerai_project import (PROJECT_ID, ROOT_COLLECTION_ID, SERVER, preflight,
                              map_legacy_inputs, notebook_inputs, restore_project_inputs)

USER = '11111111-1111-4111-8111-111111111111'
ASSET = '22222222-2222-4222-8222-222222222222'
CONTENT = '33333333-3333-4333-8333-333333333333'


class API:
    base_url = SERVER

    def __init__(self, *, flags=4095, user=USER, project=PROJECT_ID):
        self.flags, self.user, self.project = flags, user, project

    def get_json(self, method, path):
        if path == '/api/me/':
            return {'id': self.user, 'email': 'researcher@example.org'}
        if path.startswith('/api/projects/'):
            return {'id': PROJECT_ID}
        return {'owner_project_id': self.project, 'effective_flags': self.flags}


class Client:
    def __init__(self):
        self.calls = []

    def list_projects(self):
        self.calls.append('online')
        return []

    def auth_status(self):
        return SimpleNamespace(user_id=USER)


def test_preflight_confirms_online_identity_without_returning_credentials():
    client = Client()
    report = preflight(client, rest=API())
    assert report['write_checked'] and report['project_id'] == PROJECT_ID
    assert client.calls == ['online']
    assert not any('token' in key or 'credential' in key for key in report)


def test_missing_saved_login_stops_before_any_network_or_daemon_call(monkeypatch):
    monkeypatch.setattr('dataerai.rest.load_credentials', lambda: None)
    client = Client()
    with pytest.raises(RuntimeError, match='dataerai auth login'):
        preflight(client)
    assert client.calls == []


def test_saved_credentials_for_another_server_are_rejected(monkeypatch):
    monkeypatch.setattr('dataerai.rest.load_credentials',
                        lambda: SimpleNamespace(server_url='https://other.example.org'))
    client = Client()
    with pytest.raises(RuntimeError, match='requires https://beta.dataerai.com'):
        preflight(client)
    assert client.calls == []


def test_server_rejected_credentials_stop_before_daemon_or_analysis():
    from dataerai.rest import RestError

    class ExpiredLogin(API):
        def get_json(self, method, path):
            raise RestError(401, 'Authentication credentials are invalid or expired.')

    client = Client()
    with pytest.raises(RuntimeError):
        preflight(client, rest=ExpiredLogin())
    assert client.calls == []


@pytest.mark.parametrize('flags', [0, 1, 0x015])
def test_permissions_fail_before_transfer_daemon_is_used(flags):
    client = Client()
    with pytest.raises(PermissionError):
        preflight(client, rest=API(flags=flags))
    assert not client.calls


def test_wrong_server_and_identity_mismatch_stop_before_analysis():
    api = API()
    api.base_url = 'https://other.example.org'
    client = Client()
    with pytest.raises(RuntimeError, match='requires'):
        preflight(client, rest=api)
    assert not client.calls
    with pytest.raises(RuntimeError, match='different users'):
        preflight(client, rest=API(user=ASSET))


def record(path='Plumes/G1.h5', data=b'raw-data'):
    return {'path': path, 'filename': Path(path).name, 'asset_id': ASSET, 'content_id': CONTENT,
            'size_bytes': len(data), 'checksum': 'sha256:' + hashlib.sha256(data).hexdigest()}


def catalogue(root, files):
    folder = root / 'preservation'
    folder.mkdir(exist_ok=True)
    (folder / 'project-inputs.json').write_text(json.dumps(
        {'server': SERVER, 'project_id': PROJECT_ID, 'files': files}))


def test_selection_downloads_only_notebook_modalities_and_explicit_sample(tmp_path):
    files = [record(), record('AFM/10um/G1.ibw'), record('TargetMicroscopy/G1/s1.datx')]
    catalogue(tmp_path, files)
    assert notebook_inputs(tmp_path, '4_Plume_Metrics_Extraction.ipynb') == files[:1]
    assert notebook_inputs(tmp_path, '3_Structure_Characterization.ipynb') == files[1:2]
    assert notebook_inputs(tmp_path, 'quickstart.ipynb', input_paths=['Plumes/G1.h5']) == files[:1]
    with pytest.raises(FileNotFoundError):
        notebook_inputs(tmp_path, 'quickstart.ipynb', input_paths=['Plumes/missing.h5'])


@pytest.mark.parametrize('path', ['../escape.h5', '/absolute', 'Plumes/../../escape', 'Plumes\\escape'])
def test_catalogue_rejects_unsafe_paths_before_selecting_inputs(tmp_path, path):
    catalogue(tmp_path, [record(path)])
    with pytest.raises(ValueError):
        notebook_inputs(tmp_path, 'quickstart.ipynb')


def test_duplicate_paths_are_not_silently_selected(tmp_path):
    catalogue(tmp_path, [record(), record()])
    with pytest.raises(ValueError, match='Duplicate'):
        notebook_inputs(tmp_path, '4_Plume_Metrics_Extraction.ipynb')


class Downloads:
    def __init__(self, data=b'raw-data'):
        self.data, self.calls = data, []

    def download(self, asset_id, destination, **kwargs):
        self.calls.append((asset_id, kwargs))
        path = Path(destination) / 'G1.h5'
        path.write_bytes(self.data)
        return SimpleNamespace(content_id=CONTENT,
                               files=[SimpleNamespace(filename='G1.h5', local_path=str(path))])


def test_pinned_download_cache_is_revalidated_and_runs_have_independent_bytes(tmp_path):
    client, item = Downloads(), record()
    restore_project_inputs(tmp_path, client, [item], tmp_path / 'first')
    (tmp_path / 'first/data/Plumes/G1.h5').write_bytes(b'changed!')
    restore_project_inputs(tmp_path, client, [item], tmp_path / 'second')
    assert (tmp_path / 'second/data/Plumes/G1.h5').read_bytes() == b'raw-data'
    assert client.calls == [(ASSET, {'content_id': CONTENT})]
    cached = next((tmp_path / '.dataerai/project-downloads/sha256').iterdir())
    cached.write_bytes(b'corrupt!')
    restore_project_inputs(tmp_path, client, [item], tmp_path / 'third')
    assert len(client.calls) == 2


def test_corrupt_download_cannot_be_used_by_analysis(tmp_path):
    with pytest.raises(ValueError, match='checksum/size'):
        restore_project_inputs(tmp_path, Downloads(b'bad'), [record()], tmp_path / 'run')
    assert not (tmp_path / 'run/data/Plumes/G1.h5').exists()


def test_current_upstream_paths_map_to_original_input_bytes(tmp_path):
    original = tmp_path / 'data/Plumes/plume_metrics/YG063_metrics.csv'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'Sample Name,Area\nC-G6,42\n')
    coords = tmp_path / 'data/Plumes/frame_normalize_dataset/YG065_coords.npy'
    coords.parent.mkdir()
    coords.write_bytes(b'original coordinates')
    map_legacy_inputs(tmp_path)
    assert (tmp_path / 'data/datasets/ML_plume_dynamics/plume_metrics/C-G6_metrics.csv').read_bytes() == original.read_bytes()
    assert (tmp_path / 'data/datasets/ML_plume_dynamics/frame_normalize_dataset/G1_coords.npy').read_bytes() == coords.read_bytes()


def test_closed_lazy_mapping_is_not_traversed_by_provenance(tmp_path):
    """h5py handles implement Mapping but may be closed or contain gigabytes."""
    from collections.abc import Mapping
    from test_dataerai_notebook import make_tracker

    class ClosedFile(Mapping):
        def __iter__(self):
            raise AssertionError('Provenance must not traverse an HDF5 file handle')

        def __len__(self):
            raise AssertionError('Provenance must not inspect a closed file')

        def __getitem__(self, key):
            raise AssertionError('Provenance must not load HDF5 datasets')

    tracker, client, _ = make_tracker(tmp_path)
    tracker.record_cell(source='with open_hdf5() as recording: frame = read_window()',
                        assigned_names=['recording'], user_ns={'recording': ClosedFile()})
    assert tracker.cell_count == 1
    assert any(u['kwargs']['metadata']['record_kind'] == 'cell_execution' for u in client.uploads)


def test_saved_executed_notebook_has_current_outputs_and_clears_unexecuted_cells(tmp_path):
    import nbformat
    import os
    from test_dataerai_preservation import setup_repo
    from test_dataerai_notebook import FakeShell, USER_ID
    from dataerai_preservation import start_analysis
    root, path, client = setup_repo(tmp_path)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell('%%dataerai\nvalue = 7\n'),
        nbformat.v4.new_code_cell('not_executed()', execution_count=9,
            outputs=[nbformat.v4.new_output('stream',name='stdout',text='historical output')])])
    nbformat.write(notebook, path)
    cwd = Path.cwd()
    try:
        run = start_analysis(path, client=client, shell=FakeShell(), owner_type='user', owner_id=USER_ID)
        run.record_cell(source='value = 7\n', stdout='current output\n')
        run.finish()
        uploaded = next(u for u in client.uploads if u['kwargs']['metadata']['record_kind'] == 'executed_notebook')
        saved = nbformat.read(uploaded['path'], as_version=4)
        assert saved.cells[0].outputs[0].text == 'current output\n'
        assert saved.cells[0].metadata['dataerai_cell_asset_id']
        assert saved.cells[1].outputs == [] and saved.cells[1].execution_count is None
        assert saved.metadata['dataerai']['run_asset_id'] == run.run_asset_id
        assert uploaded['kwargs']['record_type'] == 'jupyter_notebook'
    finally:
        os.chdir(cwd)


def test_failed_scientific_cell_finalizes_run_before_propagating_error(tmp_path):
    import os
    from dataerai_preservation import start_analysis
    from test_dataerai_preservation import setup_repo
    from test_dataerai_notebook import FakeShell, USER_ID

    class BrokenShell(FakeShell):
        display_pub = None

        def run_cell(self, source, store_history=False):
            return SimpleNamespace(success=False, result=None, error_before_exec=None,
                                   error_in_exec=ValueError('scientific failure'))

    root, path, client = setup_repo(tmp_path)
    cwd = Path.cwd()
    try:
        run = start_analysis(path, client=client, shell=BrokenShell(), owner_type='user', owner_id=USER_ID)
        with pytest.raises(ValueError, match='scientific failure'):
            run._cell_magic('', 'raise ValueError("scientific failure")')
        assert not run.running and Path.cwd() == cwd
        assert client.metadata_updates[-1][1]['metadata']['completion']['status'] == 'completed_with_execution_errors'
        assert any(u['kwargs']['metadata']['record_kind'] == 'executed_notebook' for u in client.uploads)
    finally:
        os.chdir(cwd)
