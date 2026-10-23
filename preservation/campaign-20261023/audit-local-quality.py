import csv,hashlib,json,subprocess,zipfile,datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import imageio_ffmpeg
ROOT=Path('/home/srruo3-server-rerun/results')
QA=ROOT/'quality-review';QA.mkdir(exist_ok=True)
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'videos':[],'metric_tables':[],'arrays':[],'figure_previews':[]}
for p in sorted((ROOT/'5_Plume_Dynamic_Analysis/outputs/figures').glob('*.mp4')):
 proc=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(p),'-f','null','-'],capture_output=True,text=True)
 assert proc.returncode==0,(p,proc.stderr)
 reader=imageio_ffmpeg.read_frames(str(p));meta=next(reader);reader.close()
 row={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'complete_decode_passed':True,'video_metadata':meta};report['videos'].append(row);print('VIDEO_OK',p.name,flush=True)
assert len(report['videos'])==6
for p in sorted((ROOT/'4_Plume_Metrics_Extraction/outputs/data/datasets/ML_plume_dynamics/plume_metrics').glob('*.csv')):
 with p.open(newline='') as f:
  reader=csv.reader(f);header=next(reader);count=sum(1 for _ in reader)
 assert count>10000 and len(header)>5
 report['metric_tables'].append({'path':str(p.relative_to(ROOT)),'rows':count,'columns':header,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(report['metric_tables'])==6
for d in sorted(ROOT.iterdir()):
 if not d.is_dir() or d==QA:continue
 for p in sorted((d/'cell-artifacts').rglob('*.npz')):
  entries=[]
  with zipfile.ZipFile(p) as z:
   for n in z.namelist():
    with z.open(n) as f:
     version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
    entries.append({'member':n,'shape':list(shape),'dtype':str(dtype),'uncompressed_bytes':z.getinfo(n).file_size})
  report['arrays'].append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'entries':entries})
# Preview every exported major bitmap or TIFF without modifying result originals.
previews=[]
for d in sorted(ROOT.iterdir()):
 if not d.is_dir() or d==QA:continue
 candidates=list((d/'outputs/figures').glob('*.png'))
 stems={p.stem for p in candidates}
 candidates += [p for p in (d/'outputs/figures').glob('*.tif') if p.stem not in stems]
 for p in sorted(candidates):
  with Image.open(p) as im:
   im=im.convert('RGB');im.thumbnail((1250,950))
   out=QA/(d.name.split('_')[0]+'-'+p.stem+'.jpg');im.save(out,quality=90)
  report['figure_previews'].append({'source':str(p.relative_to(ROOT)),'preview':str(out.relative_to(ROOT))})
  previews.append(out)
# Sheets are only for visual review, not scientific output.
for offset in range(0,len(previews),6):
 group=previews[offset:offset+6];sheet=Image.new('RGB',(1800,1500),'white');draw=ImageDraw.Draw(sheet)
 for i,p in enumerate(group):
  im=Image.open(p);im.thumbnail((870,450));x=(i%2)*900;y=(i//2)*500
  draw.text((x+12,y+5),p.stem[:100],fill='black');sheet.paste(im,(x+10,y+35))
 sheet.save(QA/f'contact-{offset//6+1}.jpg',quality=90)
(QA/'local-quality-review.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print('QA_DONE',json.dumps({'videos':len(report['videos']),'metric_tables':len(report['metric_tables']),'arrays':len(report['arrays']),'figure_previews':len(previews)}),flush=True)
