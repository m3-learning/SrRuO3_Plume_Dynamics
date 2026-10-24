#!/usr/bin/env python3
"""Publish a scientific reading edition, preserving prior immutable versions."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from dataerai_preservation import connect_client,sha256,write_json
from sro_sto_plume.runtime_memory import enable_daemon_log_drain
from dataerai.rest import RestClient

def publish(edition,results):
 enable_daemon_log_drain()
 receipt_path=results/'scientific-reading-edition-publication.json'
 receipt=json.loads(receipt_path.read_text())
 for key,old in list(receipt['assets'].items()):
  if key=='SrRuO3_scientific_results.zip':path=edition.parent/key
  elif key.startswith('canonical-reading-notebooks/'):path=edition/key.split('/',1)[1]/'reading.ipynb'
  else:path=edition/key.split('/',1)[1]
  checksum=sha256(path)
  if old['sha256']==checksum and old.get('remote_download_verified'):continue
  print('Publishing',key,flush=True)
  with connect_client() as client:
   uploaded=client.update_content(old['asset_id'],str(path))
   downloaded=client.download(uploaded.asset_id,str(results/'reading-revision-receipts'/uploaded.content_id),content_id=uploaded.content_id)
  assert len(downloaded.files)==1 and sha256(downloaded.files[0].local_path)==checksum
  with RestClient() as api:
   detail=api.get_json('GET',f"/api/assets/{old['asset_id']}/")
   metadata=detail.get('metadata') or {}
   metadata.update(sha256=checksum,canonical_sample_labels=['G1','G2','G3','G4','G5','C-G6'],edition='September 27 manuscript revision')
   response=api.request('PATCH',f"/api/assets/{old['asset_id']}/",json={'metadata':metadata});response.raise_for_status()
  receipt['assets'][key]={**old,'previous_content_id':old['content_id'],'content_id':uploaded.content_id,'sha256':checksum,'remote_download_verified':True}
  write_json(receipt_path,receipt)
 print(json.dumps({'assets':len(receipt['assets']),'all_remote_download_verified':all(x.get('remote_download_verified') for x in receipt['assets'].values())}),flush=True)
 return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--edition',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args();publish(a.edition,a.results)
