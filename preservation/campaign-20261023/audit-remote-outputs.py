import concurrent.futures,datetime,hashlib,json,os,sys
from pathlib import Path
os.environ['PATH']='/home/srruo3-venv/bin:/home/tools:'+os.environ['PATH']
sys.path.insert(0,'/home/SrRuO3_Plume_Dynamics')
from dataerai.rest import RestClient
from dataerai_project import verify_file
R=Path('/home/srruo3-server-rerun/results');v=json.loads((R/'validation.json').read_text());assert v['full_workflow_complete'] and not v['failures'] and v['server_only_inputs'] and len(v['notebooks'])==9
inventory={'schema':'org.dataerai.srruo3-final-audit/v1','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'notebooks':[],'output_assets':[],'failures':[]}
jobs=[]
for row in v['notebooks']:
 d=R/Path(row['notebook']).stem
 saved=next((d/'dataerai-saved').rglob('*.ipynb'));n=json.loads(saved.read_text())
 cells=[c['metadata']['dataerai_cell_asset_id'] for c in n['cells'] if c.get('execution_count') is not None]
 inventory['notebooks'].append({'notebook':row['notebook'],'run_asset_id':row['run_asset_id'],'captured_cells':len(cells),'executed_notebook_asset_id':row['executed_notebook_asset_id'],'source_base_verified':row['base_sha256_verified'],'input_content_versions':row['input_content_versions']})
 for cid in cells:jobs.append((d,cid))
def graph(job):
 d,cid=job
 with RestClient() as a:edges=a.list_relationships(cid)
 return d,cid,[e['related_asset'] for e in edges if e['type']=='generated_by' and e['direction']=='incoming']
assetjobs={}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for d,cid,items in pool.map(graph,jobs):
  for x in items:assetjobs[x['id']]=(d,cid,x)
print('REMOTE_GRAPH_OK',len(jobs),'cells',len(assetjobs),'outputs',flush=True)
def inspect(job):
 aid,(d,cid,x)=job
 with RestClient() as a:detail=a.get_json('GET',f'/api/assets/{aid}/')
 c=detail['current_content'];assert c['status']=='available'
 fs=c['files'];assert len(fs)==1 and fs[0]['uploaded'];f=fs[0]
 m=detail.get('metadata',{});kind=m.get('record_kind');p=None
 if kind=='saved_file':
  rel=m['path'];p=d/'outputs'/rel
 else:
  seq=m.get('run',{}).get('cell_sequence');p=d/'cell-artifacts'/f'cell-{seq:04d}'/f['filename'] if seq is not None else None
 assert p and p.is_file(),(aid,kind,m,p)
 assert verify_file(p,{'size_bytes':f['size'],'checksum':f['checksum']}),(aid,p)
 return {'notebook':d.name,'cell_asset_id':cid,'asset_id':aid,'content_id':c['id'],'path':str(p.relative_to(R)),'size_bytes':f['size'],'checksum':f['checksum'],'record_kind':kind,'remote_available':True,'local_bytes_match_remote':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for index,item in enumerate(pool.map(inspect,assetjobs.items()),1):
  inventory['output_assets'].append(item)
  if index%50==0:print('REMOTE_HASH_OK',index,flush=True)
for name in ['4_Plume_Metrics_Extraction.ipynb','5_Plume_Dynamic_Analysis.ipynb']:
 row=next(x for x in v['notebooks'] if x['notebook']==name);pins={x.get('path'):x for x in row['input_content_versions']}
 assert pins['Plumes/G2.h5']['asset_id']=='b727c17c-7d68-4a58-8558-7d72eda03d08'
 assert pins['Plumes/G2.h5']['content_id']=='1f220b45-adf6-4f78-941c-3b1cb6987288'
 assert pins['Plumes/G4.h5']['asset_id']=='58b33cc2-4f36-44d3-982b-76ab08fbc457'
 assert pins['Plumes/G4.h5']['content_id']=='859bec48-a08a-4c50-a666-a2fad6a1e11f'
mp4=[x for x in inventory['output_assets'] if x['path'].endswith('.mp4')];assert len(mp4)==6
metric=[x for x in inventory['output_assets'] if x['path'].endswith('_metrics.csv') and x['record_kind']=='saved_file'];assert len(metric)==6
inventory['all_output_assets_verified']=True;inventory['corrected_G2_G4_pins_verified']=True
(R/'quality-review/remote-output-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print('REMOTE_AUDIT_DONE',len(inventory['output_assets']),flush=True)
