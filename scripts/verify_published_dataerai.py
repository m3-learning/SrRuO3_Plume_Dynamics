#!/usr/bin/env python3
"""Execute downloaded base notebooks and verify their saved Dataerai provenance.

Requires a prepared Python 3.11 kernel and an authenticated beta login. This
explicit live check creates real notebook-run records and downloads raw input.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import uuid

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataerai_preservation import connect_client, sha256, write_json
from dataerai_project import preflight, verify_file


def verify(names, kernel):
    from dataerai.rest import RestClient
    publication = json.loads((ROOT / 'preservation/notebook-publication.json').read_text())
    folder = ROOT / '.dataerai/published-validation' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    report_path = ROOT / 'preservation/project-validation.json'
    report = {'publication_collection_id': publication['collection_id'], 'notebooks': [],
              'scientific_reexecution': 'Only the notebook names listed here were executed in full.'}
    with connect_client() as client:
        preflight(client, collection_id=publication['collection_id'])
        for name in names:
            entry = publication['assets']['notebooks/' + name]
            result = client.download(entry['asset_id'], str(folder), content_id=entry['content_id'])
            if len(result.files) != 1 or result.content_id != entry['content_id']:
                raise AssertionError('Published notebook did not match its pinned content version.')
            path = Path(result.files[0].local_path).resolve()
            assert path.is_relative_to(folder.resolve()) and sha256(path) == entry['sha256']
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            assert all(not cell.outputs and cell.execution_count is None
                       for cell in notebook.cells if cell.cell_type == 'code')
            print('EXECUTING PUBLISHED NOTEBOOK', name, flush=True)
            try:
                NotebookClient(notebook, timeout=1800, kernel_name=kernel,
                    resources={'metadata': {'path': str(folder)}}).execute()
            finally:
                nbformat.write(notebook, folder / ('executed-' + name))
            streams = '\n'.join(output.text for cell in notebook.cells
                                for output in cell.get('outputs', []) if output.output_type == 'stream')
            run_id = re.search(r'run asset:\s+([0-9a-f-]{36})', streams).group(1)
            with RestClient() as api:
                run = api.get_json('GET', f'/api/assets/{run_id}/')
                metadata = run['metadata']
                assert metadata['completion']['status'] == 'completed'
                edges = api.list_relationships(run_id)
                cells = [edge for edge in edges if edge['type'] == 'part_of_run' and edge['direction'] == 'incoming']
                assert len(cells) == metadata['completion']['cell_count']
                versions = metadata['environment']['input_content_versions']
                if name.startswith('00_'):
                    assert any(item.get('path') == 'Plumes/G5.h5' and item.get('content_id') for item in versions)
                executed = next(edge['related_asset']['id'] for edge in edges
                                if edge['type'] == 'generated_by' and edge['direction'] == 'incoming')
                detail = api.get_json('GET', f'/api/assets/{executed}/')
                content = detail['current_content']
                file = content['files'][0]
            destination = folder / 'saved-notebooks' / run_id
            downloaded = client.download(executed, str(destination), content_id=content['id'])
            assert len(downloaded.files) == 1
            saved = Path(downloaded.files[0].local_path)
            assert saved.resolve().is_relative_to(destination.resolve())
            assert verify_file(saved, {'size_bytes': file['size'], 'checksum': file['checksum']})
            restored = nbformat.read(saved, as_version=4)
            nbformat.validate(restored)
            assert restored.metadata['dataerai']['run_asset_id'] == run_id
            captured = [cell for cell in restored.cells if cell.get('execution_count') is not None]
            assert len(captured) == len(cells)
            assert all(cell.metadata.get('dataerai_cell_asset_id') for cell in captured)
            assert not any(output.output_type == 'error' for cell in restored.cells
                           for output in cell.get('outputs', []))
            record = {'notebook': name, 'base_asset_id': entry['asset_id'],
                      'base_content_id': entry['content_id'], 'run_asset_id': run_id,
                      'executed_notebook_asset_id': executed, 'executed_notebook_content_id': content['id'],
                      'summary_asset_id': metadata['completion']['summary_asset_id'],
                      'captured_cells': len(cells), 'asset_count': metadata['completion']['asset_count'],
                      'relationship_count': metadata['completion']['relationship_count'],
                      'input_content_versions': versions, 'status': 'passed'}
            report['notebooks'].append(record)
            write_json(report_path, report)
            print('PASSED', name, json.dumps(record, default=str), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--kernel', default='python3')
    parser.add_argument('--notebook', action='append', dest='names')
    args = parser.parse_args()
    verify(args.names or ['0_Dataerai_Preservation.ipynb', '00_Dataerai_Raw_Data_Quickstart.ipynb',
                         '7_Target_Spatial_Inhomogeneity.ipynb'], args.kernel)
