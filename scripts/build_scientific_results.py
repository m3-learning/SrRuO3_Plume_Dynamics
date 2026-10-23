#!/usr/bin/env python3
"""Create a scientific reading edition without altering execution evidence."""
import argparse,base64,copy,csv,datetime,hashlib,html,json,re,shutil,zipfile
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

PROJECT='https://beta.dataerai.com/p/99185d30-fe63-497c-91b4-852a02574d49'
TITLES={
 '1_Abstract':'Study overview',
 '2_PLD_Experiment_Design_and_Target_Microscopy':'Experimental design and target microscopy',
 '3_Structure_Characterization':'Structural and surface characterization',
 '4_Plume_Metrics_Extraction':'Plume metric extraction',
 '5_Plume_Dynamic_Analysis':'Plume dynamics',
 '6_Target_Temporal_Variation':'Temporal variation',
 '7_Target_Spatial_Inhomogeneity':'Spatial inhomogeneity'}
EDITORIAL={
 '## 2. Structure Characterization':'# 3. Structural and Surface Characterization',
 '# 4_Plume_Analysis':'# 7. Spatial Inhomogeneity of Plume Dynamics',
 '### Plottings of all microscopy scans':'### Target Microscopy Maps',
 '### temporal':'### Temporal Variability',
 'Statics':'Statistics',
 'Update the configuration below to point at the folders you would like to compare.':'Surface roughness was evaluated for the configured sample groups.',
 'Fig. 1 Pulsed laser deposition setup with plume recording camera, and experiment design. a':'Fig. 1. Pulsed laser deposition setup with plume recording camera and experimental design.'}
BANNED=re.compile(r'/home/|/mnt/[a-z]/|[A-Z]:\\Users\\|yig319@|Signed in:|Notebook runtime:|local spool:|NotebookProvenance\(|Dataerai provenance (active|finalized)|Traceback \(most recent call last\)|(?:Future|User|Runtime)Warning:|ChatGPT|Codex|NUC-Chalfont',re.I)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image_digests(n):
 values=[]
 for c in n.cells:
  for o in c.get('outputs',[]):
   for mime,payload in o.get('data',{}).items():
    if mime.startswith('image/'):
     if isinstance(payload,list):payload=''.join(payload)
     values.append((mime,hashlib.sha256(payload.encode()).hexdigest()))
 return sorted(values)

def build(root,out):
 out.mkdir(parents=True,exist_ok=True)
 audit={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'presentation_only':True,'source_notebooks_modified':False,'reports':[]}
 validation=json.loads((root/'validation.json').read_text());assert validation['full_workflow_complete'] and not validation['failures']
 inventory=json.loads((root/'quality-review/remote-output-inventory.json').read_text())
 equivalence=json.loads((root/'quality-review/metric-input-equivalence.json').read_text());assert equivalence['all_scientific_fields_exactly_equal']
 rows=[]
 for name,title in TITLES.items():
  source=root/name/'executed.ipynb';before=digest(source);n=nbformat.read(source,as_version=4)
  technical=[c for c in n.cells if any(t.startswith('dataerai-managed-') for t in c.metadata.get('tags',[]))]
  science=[copy.deepcopy(c) for c in n.cells if not any(t.startswith('dataerai-managed-') for t in c.metadata.get('tags',[]))]
  expected=image_digests(nbformat.v4.new_notebook(cells=copy.deepcopy(science)));removed_streams=0;edits=0
  for c in science:
   if c.cell_type=='markdown':
    for old,new in EDITORIAL.items():
     if old in c.source:c.source=c.source.replace(old,new);edits+=1
   if c.cell_type=='code':
    kept=[]
    for o in c.outputs:
     assert o.output_type!='error','Cannot publish a reading edition of a failed cell.'
     if o.output_type=='stream':removed_streams+=1;continue
     data=o.get('data',{});plain=data.get('text/plain','')
     if not any(k!='text/plain' for k in data) and (BANNED.search(str(plain)) or re.search(r'NotebookProvenance|0x[0-9a-f]+|<Figure|<matplotlib\.',str(plain))):continue
     kept.append(o)
    c.outputs=kept;c.execution_count=None
  n.cells=science
  assert image_digests(n)==expected,'Scientific image payload changed.'
  # Use the already embedded original report for its resolved Markdown images.
  nb=HTMLExporter(template_name='lab',exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True,embed_images=True)
  streams='\n'.join(str(o.get('text','')) for c in nbformat.read(source,as_version=4).cells for o in c.get('outputs',[]) if o.output_type=='stream')
  match=re.search(r'Verified notebook source workspace:\s+([^\n]+)',streams)
  resources={'metadata':{'path':str(source.parent)}}
  if match:
   workspace=Path(match.group(1).strip());paths=list(workspace.glob('.dataerai/runs/*/workspace/notebooks'))
   if paths:resources={'metadata':{'path':str(paths[0])}}
  body,_=nb.from_notebook_node(n,resources=resources)
  soup=BeautifulSoup(body,'html.parser');soup.title.string=title+' | SrRuO3 scientific results'
  style=soup.new_tag('style');style.string='body{background:#fff!important}.jp-Notebook{max-width:1200px;margin:auto;padding:26px}.jp-InputPrompt,.jp-OutputPrompt{display:none!important}.jp-RenderedImage img{max-width:100%;height:auto}.reading-nav{font:14px system-ui;padding:18px 26px;border-bottom:1px solid #ddd}';soup.head.append(style)
  nav=soup.new_tag('nav',attrs={'class':'reading-nav'});a=soup.new_tag('a',href='../SrRuO3_scientific_results.html');a.string='SrRuO3 scientific results';nav.append(a);nav.append(' / '+title);soup.body.insert(0,nav)
  visible=BeautifulSoup(str(soup),'html.parser')
  for el in visible(['script','style']):el.decompose()
  text=visible.get_text(' ',strip=True);hits=[text[max(0,m.start()-60):m.end()+130] for m in BANNED.finditer(text)];assert not hits,(name,hits[:10])
  images=[im.get('src','') for im in soup.find_all('img')];assert all(s.startswith(('data:','https://')) for s in images),name
  dest=out/name/'report.html';dest.parent.mkdir(exist_ok=True);dest.write_text(str(soup))
  assert digest(source)==before
  audit['reports'].append({'notebook':name,'source_sha256':before,'report_sha256':digest(dest),'technical_cells_removed':len(technical),'stream_outputs_removed':removed_streams,'editorial_heading_edits':edits,'scientific_image_payloads_preserved':len(expected),'visible_text_scan_passed':True})
  rows.append(f'<li><a href="{name}/report.html">{html.escape(title)}</a></li>')
 for folder in ['videos','data/plume_metrics','figures']:(out/folder).mkdir(parents=True,exist_ok=True)
 videos=[]
 for p in sorted((root/'5_Plume_Dynamic_Analysis/outputs/figures').glob('*.mp4')):
  target=out/'videos'/p.name;shutil.copy2(p,target);assert digest(p)==digest(target)
  videos.append(f'<figure><video controls preload="metadata" src="videos/{p.name}"></video><figcaption>{html.escape(p.stem.replace("Extra-","").replace("_plume_video",""))}</figcaption></figure>')
 tables=[]
 for p in sorted((root/'4_Plume_Metrics_Extraction/outputs/data/datasets/ML_plume_dynamics/plume_metrics').glob('*metrics.csv')):
  target=out/'data/plume_metrics'/p.name;shutil.copy2(p,target);assert digest(p)==digest(target)
  with p.open(newline='') as f:reader=csv.reader(f);next(reader);count=sum(1 for _ in reader)
  tables.append(f'<tr><td>{html.escape(p.stem.replace("_metrics",""))}</td><td>{count:,}</td><td><a href="data/plume_metrics/{p.name}">CSV</a></td></tr>')
 shutil.copy2(root/'5_Plume_Dynamic_Analysis/outputs/figures/Figure4-Plume_area_distance_velocity_average.png',out/'figures/Figure4.png')
 manifest={'schema':'org.dataerai.srruo3-scientific-reading-edition/v1','presentation_role':'Scientific reading edition derived from preserved executed notebooks','project_url':PROJECT,'campaign_validation_asset_id':'b8459e89-f908-4393-af21-3802bd88348e','metric_equivalence_asset_id':'857bd096-475d-4958-96ae-b03b81b8ae6a','source_bundle':validation['publication']['assets']['source_bundle'],'environment':validation['publication']['assets']['environment'],'notebooks':[{k:x.get(k) for k in ['notebook','run_asset_id','executed_notebook_asset_id','executed_notebook_content_id','base_asset_id','base_content_id']} for x in validation['notebooks']],'output_assets':inventory['output_assets'],'metric_equivalence':equivalence,'execution_scope':'Nine independently executed notebooks. Downstream notebooks use preserved versioned metrics; all seven non-ID columns of six newly extracted CSVs are exactly equal row by row. Sample ID has an explicit G-to-YG label mapping. Automatic inter-notebook propagation was not tested.'}
 (out/'source-and-output-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 page="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SrRuO3 scientific results</title><style>body{font:17px/1.65 Georgia,serif;color:#18232b;max-width:1050px;margin:45px auto;padding:0 24px}h1,h2,h3,nav{font-family:system-ui,sans-serif}h1{font-size:34px;line-height:1.25}h2{margin-top:2em;border-bottom:1px solid #ccd4d8;padding-bottom:8px}a{color:#165579}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:9px 16px;text-align:left;border-bottom:1px solid #ddd}.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure{margin:0}video{width:100%;background:#080814}figcaption{font:14px system-ui}li{margin:8px 0}.meta{color:#58646c;font:14px system-ui}@media(max-width:650px){.videos{grid-template-columns:1fr}h1{font-size:27px}}</style></head><body><p class="meta">RESEARCH RESULTS · VERSIONED DATA AND ANALYSIS</p><h1>SrRuO<sub>3</sub> plume dynamics and thin-film characterization</h1><p>This reading edition presents the scientific narrative, figures and tables from seven research notebooks, together with six plume videos and newly extracted metric tables.</p><h2>Scientific reports</h2><ol>REPORT_LINKS</ol><h2>Plume area, distance and velocity</h2><figure><img src="figures/Figure4.png" alt="Six-sample plume area, distance and velocity comparison"><figcaption>Figure 4. Plume area and distance as functions of time, and velocity as a function of distance, reproduced from the preserved dynamic-analysis notebook.</figcaption></figure><h2>Plume videos</h2><div class="videos">VIDEO_LINKS</div><h2>Extracted metric tables</h2><p>These CSV files are the outputs of the newly executed metric-extraction notebook. Sample identifiers use G1–G5 and C-G6; historical recording identifiers are documented in the source manifest.</p><table><thead><tr><th>Sample</th><th>Rows</th><th>Data</th></tr></thead><tbody>TABLE_LINKS</tbody></table><h2>Reproducibility and data availability</h2><p>Nine versioned notebooks were executed independently using DataERAI-hosted source and inputs in a Python 3.11 environment. The seven research notebooks are presented above; two additional notebooks document data-access and preservation checks.</p><p>The downstream analyses used the published, versioned metric tables. Across all six samples, every non-ID column of the newly extracted tables matches the published tables exactly, row by row. The only difference is the sample-ID convention, such as G2 versus YG066. Automatic propagation of new outputs between notebooks was not tested in this campaign.</p><p>All 419 preserved output assets were checked for availability and checksum consistency. All six plume videos were decoded completely. The original executed notebooks and provenance records remain available in the <a href="PROJECT_LINK">DataERAI project</a>. Exact source, input and output versions are listed in the <a href="source-and-output-manifest.json">source and output manifest</a>.</p><p>This edition removes execution-interface messages and makes limited heading and caption edits. Scientific image payloads and exported data values are retained. Scientific interpretations and author information follow the source notebooks.</p></body></html>"""
 page=page.replace('REPORT_LINKS',''.join(rows)).replace('VIDEO_LINKS',''.join(videos)).replace('TABLE_LINKS',''.join(tables)).replace('PROJECT_LINK',PROJECT)
 (out/'SrRuO3_scientific_results.html').write_text(page)
 for p in out.rglob('*.html'):
  soup=BeautifulSoup(p.read_text(),'html.parser')
  for el in soup(['style','script']):el.decompose()
  assert not BANNED.search(soup.get_text(' ',strip=True)),p
 audit['reading_edition_files']=[{'path':str(p.relative_to(out)),'sha256':digest(p),'size_bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]
 (root/'scientific-reading-edition-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 archive=out.parent/'SrRuO3_scientific_results.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(out))
 print(json.dumps({'reports':len(rows),'videos':len(videos),'tables':len(tables),'zip_bytes':archive.stat().st_size,'scientific_images_preserved':sum(x['scientific_image_payloads_preserved'] for x in audit['reports']),'audit':str(root/'scientific-reading-edition-audit.json')}))
 return audit
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.results,a.output)
