import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import uuid

import nbformat
from nbclient import NotebookClient

ROOT=Path('/home/SrRuO3_Plume_Dynamics')
sys.path.insert(0,str(ROOT))
from dataerai_preservation import connect_client, sha256, write_json
from dataerai_project import preflight, verify_file
from dataerai.rest import RestClient

RUN_ROOT=Path('/home/srruo3-server-rerun')
RUN_ROOT.mkdir(parents=True,exist_ok=True)
OUT=RUN_ROOT/'results'
OUT.mkdir(exist_ok=True)
publication=json.loads((ROOT/'preservation/notebook-publication.json').read_text())

report_path=OUT/'validation.json'
report=json.loads(report_path.read_text()) if report_path.exists() else {'notebooks':[], 'failures':[], 'server_only_inputs':True, 'scientific_cells_excluded':False}
report['publication']=publication
report.setdefault('failures',[])
shared=RUN_ROOT/'cache'
for category in ['project-downloads','downloads']:
    (shared/category).mkdir(parents=True,exist_ok=True)
os.environ['SRRUO3_SHARED_CACHE']=str(shared)
folder=RUN_ROOT
names=sys.argv[1:] or ['0_Dataerai_Preservation.ipynb','00_Dataerai_Raw_Data_Quickstart.ipynb',
    '1_Abstract.ipynb','2_PLD_Experiment_Design_and_Target_Microscopy.ipynb',
    '3_Structure_Characterization.ipynb','4_Plume_Metrics_Extraction.ipynb',
    '5_Plume_Dynamic_Analysis.ipynb','6_Target_Temporal_Variation.ipynb',
    '7_Target_Spatial_Inhomogeneity.ipynb']
force_names=set()
sys.path.insert(0,str(ROOT/'src'))
from sro_sto_plume.runtime_memory import enable_daemon_log_drain
enable_daemon_log_drain()

def collect(notebook, path, name):
    destination=OUT/Path(name).stem
    destination.mkdir(exist_ok=True)
    nbformat.write(notebook,destination/'executed.ipynb')
    streams='\n'.join(o.text for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='stream')
    match=re.search(r'Verified notebook source workspace:\s+([^\n]+)',streams)
    if match:
        workspace=Path(match.group(1).strip())
        for run in (workspace/'.dataerai/runs').glob('*'):
            for p in run.glob('*.json'):
                shutil.copy2(p,destination/p.name)
            for p in sorted((run/'files').rglob('*')):
                if p.is_file():
                    relative=p.relative_to(run/'files')
                    target=destination/'outputs'/Path(*relative.parts[1:])
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(p,target)
            for p in run.glob('cell-*/*'):
                if p.is_file():
                    target=destination/'cell-artifacts'/p.parent.name/p.name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        target.unlink()
                    os.link(p,target)
    return streams,destination

with connect_client() as client:
    preflight(client,collection_id=publication['collection_id'])
    for name in names:
        if name not in force_names and any(r['notebook']==name and r.get('status')=='passed' for r in report['notebooks']):
            print('ALREADY PASSED',name,flush=True)
            continue
        notebook=None
        try:
            entry=publication['assets']['notebooks/'+name]
            staging=folder/'base'/Path(name).stem
            staging.mkdir(parents=True)
            result=client.download(entry['asset_id'],str(staging),content_id=entry['content_id'])
            assert len(result.files)==1 and result.content_id==entry['content_id']
            original_path=Path(result.files[0].local_path).resolve()
            assert original_path.is_relative_to(staging.resolve()) and sha256(original_path)==entry['sha256']
            notebook=nbformat.read(original_path,as_version=4)
            assert notebook.metadata['dataerai']['source_bundle']==publication['assets']['source_bundle']
            assert notebook.metadata['dataerai']['environment']==publication['assets']['environment']
            path=folder/name
            shutil.copyfile(original_path,path)
            assert sha256(path)==entry['sha256']
            nbformat.validate(notebook)
            print('EXECUTING',name,flush=True)
            def progress(cell,cell_index,**kwargs):
                print('CELL START',name,cell_index,flush=True)
            try:
                NotebookClient(notebook,timeout=None,kernel_name='srruo3',
                    resources={'metadata':{'path':str(folder)}},on_cell_start=progress).execute()
            finally:
                nbformat.write(notebook,folder/('executed-'+name))
                streams,destination=collect(notebook,path,name)
            run_id=re.search(r'run asset:\s+([0-9a-f-]{36})',streams).group(1)
            with RestClient() as api:
                run=api.get_json('GET',f'/api/assets/{run_id}/')
                metadata=run['metadata']
                assert metadata['completion']['status']=='completed'
                if name.startswith(('4_','5_')):
                    pinned={r['path']:r for r in json.loads((ROOT/'preservation/project-inputs.json').read_text())['files']}
                    used={r['path']:r for r in metadata['environment']['input_content_versions'] if 'path' in r}
                    for required in ['Plumes/G2.h5','Plumes/G4.h5']:
                        assert used[required]['asset_id']==pinned[required]['asset_id']
                        assert used[required]['content_id']==pinned[required]['content_id']

                edges=api.list_relationships(run_id)
                cells=[e for e in edges if e['type']=='part_of_run' and e['direction']=='incoming']
                assert len(cells)==metadata['completion']['cell_count']
                executed=next(e['related_asset']['id'] for e in edges if e['type']=='generated_by' and e['direction']=='incoming')
                detail=api.get_json('GET',f'/api/assets/{executed}/')
                content=detail['current_content']; file=content['files'][0]
            downloaded=client.download(executed,str(destination/'dataerai-saved'/run_id),content_id=content['id'])
            saved=Path(downloaded.files[0].local_path)
            assert verify_file(saved,{'size_bytes':file['size'],'checksum':file['checksum']})
            restored=nbformat.read(saved,as_version=4)
            nbformat.validate(restored)
            assert restored.metadata['dataerai']['run_asset_id']==run_id
            captured=[c for c in restored.cells if c.get('execution_count') is not None]
            assert len(captured)==len(cells)
            assert all(c.metadata.get('dataerai_cell_asset_id') for c in captured)
            assert not any(o.output_type=='error' for c in restored.cells for o in c.get('outputs',[]))
            record={'notebook':name,'base_asset_id':entry['asset_id'],'base_content_id':entry['content_id'],
                'executed_source_sha256':sha256(path),'run_asset_id':run_id,
                'executed_notebook_asset_id':executed,'executed_notebook_content_id':content['id'],
                'summary_asset_id':metadata['completion']['summary_asset_id'],'captured_cells':len(cells),
                'asset_count':metadata['completion']['asset_count'],
                'relationship_count':metadata['completion']['relationship_count'],
                'input_content_versions':metadata['environment']['input_content_versions'],'status':'passed'}
            if name in force_names:
                report.setdefault('superseded_successful_runs',[]).extend(
                    {**r,'superseded_reason':'Re-executed after rendering review to preserve composite panels and inline-only plots.'}
                    for r in report['notebooks'] if r['notebook']==name)
            report['notebooks']=[r for r in report['notebooks'] if r['notebook']!=name]+[record]
            report['failures']=[r for r in report['failures'] if r['notebook']!=name]
            record['base_sha256_verified']=sha256(path)==entry['sha256']
            record['scientific_cells_excluded']=False
            # Remote inputs and saved artifacts were verified before removing only
            # derived raw copies in this completed run's isolated workspace.
            workspace_match=re.search(r'Verified notebook source workspace:\s+([^\n]+)',streams)
            cleaned=[]
            if workspace_match:
                source_workspace=Path(workspace_match.group(1).strip()).resolve()
                assert source_workspace.is_relative_to(RUN_ROOT.resolve())
                for raw in source_workspace.glob('.dataerai/runs/*/workspace/data/Plumes/*.h5'):
                    assert raw.resolve().is_relative_to(RUN_ROOT.resolve()) and not raw.is_symlink()
                    if raw.name in {'G1.h5','G2.h5','G3.h5','G4.h5','G5.h5','C-G6.h5'}:
                        cleaned.append({'path':str(raw),'size_bytes':raw.stat().st_size})
                        raw.unlink()
            (destination/'local-input-cleanup.json').write_text(json.dumps(cleaned,indent=2))
            from nbconvert import HTMLExporter
            resources={'metadata':{'path':str(destination)}}
            if workspace_match:
                run_workspaces=list(source_workspace.glob('.dataerai/runs/*/workspace/notebooks'))
                if run_workspaces: resources={'metadata':{'path':str(run_workspaces[0])}}
            html,_=HTMLExporter(template_name='lab',exclude_input=True,embed_images=True).from_notebook_node(notebook,resources=resources)
            (destination/'report.html').write_text(html)
            print('PASSED',name,'run',run_id,'raw_copies_removed',len(cleaned),flush=True)
        except Exception as error:
            print('FAILED',name,type(error).__name__,flush=True)
            traceback.print_exc()
            report['failures'].append({'notebook':name,'exception_type':type(error).__name__,
                'execution_folder':str(folder),'local_notebook':str(folder/('executed-'+name))})
        write_json(report_path,report)
print('FINISHED',json.dumps({'passed':[r['notebook'] for r in report['notebooks']],
    'failed':[r['notebook'] for r in report['failures']]}),flush=True)

expected=9
report['full_workflow_complete']=len(report['notebooks'])==expected and not report['failures']
write_json(report_path,report)
lines=['# Server-only SrRuO3 full workflow validation','',
       'All notebook sources, runtime code, environments, and inputs were downloaded from immutable DataERAI versions. No local input originals or selected-cell exclusions were used.','',
       '| Notebook | Result | Report |','|---|---|---|']
for row in report['notebooks']:
    stem=Path(row['notebook']).stem
    lines.append('| '+row['notebook']+' | Passed; remote artifacts verified | [View]('+stem+'/report.html) |')
for row in report['failures']:
    lines.append('| '+row['notebook']+' | Failed: '+row['exception_type']+' | |')
lines.extend(['','Full workflow complete: '+str(report['full_workflow_complete']),
              '','[Validation receipts](validation.json). Each result folder contains the executed notebook, exact input/run records, figures, tables, arrays and rich displays. Full-resolution plume arrays remain full resolution; per-run raw input copies are removed only after remote validation. The checksum-addressed cache is reusable.'])
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
from dataerai_preservation import record_type,link
with connect_client() as client:
    result=client.upload(str(report_path),title='SrRuO3 full workflow validation - '+('PASSED' if report['full_workflow_complete'] else 'INCOMPLETE'),
        owner_type='project',owner_id=publication['project_id'],collection_id=publication['run_collection_id'],
        record_type=record_type(report_path),tags=['srruo3-plume-dynamics','full-workflow-validation'],
        metadata={'schema':'org.dataerai.srruo3-full-validation/v1','full_workflow_complete':report['full_workflow_complete'],
                  'passed_notebooks':len(report['notebooks']),'failed_attempts':len(report['failures']),
                  'server_only_inputs':True,'sha256':sha256(report_path)})
    for row in report['notebooks']:
        link(client,result.asset_id,row['run_asset_id'],'summarizes')
    downloaded=client.download(result.asset_id,str(OUT/'campaign-receipt-download'),content_id=result.content_id)
    assert sha256(downloaded.files[0].local_path)==sha256(report_path)
    write_json(OUT/'campaign-publication.json',{'asset_id':result.asset_id,'content_id':result.content_id,
        'sha256':sha256(report_path),'remote_download_verified':True})
print('CAMPAIGN_FINISHED',report['full_workflow_complete'],flush=True)
