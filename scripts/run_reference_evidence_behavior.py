#!/usr/bin/env python3
"""Test predefined evidence deletion against matched neighbor/random controls."""
from __future__ import annotations

import argparse, hashlib, json, random, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter
from run_option_conditioned_visual_evidence import OptionEvidenceModel, OPTION_LETTERS, margins, model_prompt, sha256_file

REASONING=("{question}\nGive concise image-grounded reasoning inside <think>...</think>. "
           "Then output exactly one option letter (A, B, C, or D) inside <answer>...</answer>.")

def parsed(text):
 m=re.findall(r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)",text,re.I|re.S); c=m[-1] if m else text
 x=re.search(r"(?:^|\s|\()([ABCD])(?:\)|[\s.,:;\-]|$)",c,re.I); return x.group(1).upper() if x else None
def translate(mask,dx,dy):
 h,w=mask.shape; out=np.zeros_like(mask); sx0=max(0,-dx); sx1=min(w,w-dx); sy0=max(0,-dy); sy1=min(h,h-dy)
 out[sy0+dy:sy1+dy,sx0+dx:sx1+dx]=mask[sy0:sy1,sx0:sx1]; return out
def iou(a,b):
 u=np.logical_or(a,b).sum(); return float(np.logical_and(a,b).sum()/u) if u else 0.0
def controls(mask,seed,count=3):
 ys,xs=np.where(mask); h,w=mask.shape; bw=xs.max()-xs.min()+1; bh=ys.max()-ys.min()+1; area=int(mask.sum())
 candidates=[]
 for dy,dx in [(0,bw),(0,-bw),(bh,0),(-bh,0),(bh,bw),(-bh,-bw),(bh,-bw),(-bh,bw)]:
  x=translate(mask,dx,dy)
  if int(x.sum())==area and iou(x,mask)<0.15: candidates.append(x)
 neighbor=candidates[0] if candidates else np.roll(mask,(max(1,bh),max(1,bw)),axis=(0,1))
 rng=random.Random(seed); randoms=[]; seen=set()
 minx,maxx=int(xs.min()),int(xs.max()); miny,maxy=int(ys.min()),int(ys.max())
 for _ in range(10000):
  nx=rng.randint(0,max(0,w-bw)); ny=rng.randint(0,max(0,h-bh)); dx=nx-minx; dy=ny-miny
  if (dx,dy) in seen: continue
  seen.add((dx,dy)); x=translate(mask,dx,dy)
  if int(x.sum())==area and iou(x,mask)<0.15: randoms.append(x)
  if len(randoms)==count: break
 if len(randoms)<count: randoms.extend([np.roll(mask,(rng.randrange(h),rng.randrange(w)),axis=(0,1)) for _ in range(count-len(randoms))])
 return neighbor,randoms
def delete(image,base,mask):
 a=np.asarray(image).copy(); b=np.asarray(base); a[mask]=b[mask]; return Image.fromarray(a)
def reasoning_prompt(processor,q):
 return processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":REASONING.format(question=q)}]}],tokenize=False,add_generation_prompt=True)
def generate(model,prompt,image):
 x=model.processor(text=[prompt],images=[image],return_tensors='pt'); x={k:v.to('cuda') for k,v in x.items()}
 with torch.inference_mode(): y=model.model.generate(**x,max_new_tokens=1024,do_sample=False,use_cache=True)
 return model.processor.tokenizer.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--panel',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--limit',type=int); ap.add_argument('--start',type=int,default=0); ap.add_argument('--stop',type=int); a=ap.parse_args()
 if a.output_dir.exists(): raise FileExistsError(a.output_dir)
 panel=json.loads(a.panel.read_text()); cases=panel['cases'][a.start:a.stop]; cases=cases[:a.limit]; a.output_dir.mkdir(parents=True)
 model=OptionEvidenceModel(a.model,batch_size=8); rows=[]
 with (a.output_dir/'case_results.jsonl').open('w') as f:
  for case in cases:
   path=Path(case['image']);
   if sha256_file(path)!=case['image_sha256']: raise ValueError('image hash')
   image=Image.open(path).convert('RGB'); w,h=image.size; mask=np.zeros((h,w),bool)
   for x0,y0,x1,y1 in case['external_annotation']['boxes']:
    l=max(0,min(w-1,round(x0*w/1000))); t=max(0,min(h-1,round(y0*h/1000))); r=max(l+1,min(w,round(x1*w/1000))); b=max(t+1,min(h,round(y1*h/1000))); mask[t:b,l:r]=True
   seed=int(hashlib.sha256(case['source_record_sha256'].encode()).hexdigest()[:16],16); neighbor,randoms=controls(mask,seed)
   base=image.filter(ImageFilter.GaussianBlur(radius=max(2.0,min(w,h)*.035)))
   images={'clean':image,'reference_deleted':delete(image,base,mask),'neighbor_deleted':delete(image,base,neighbor)}
   images.update({f'random_deleted_{i}':delete(image,base,x) for i,x in enumerate(randoms)})
   forced=model_prompt(model.processor,case['problem']); scored=model.score(forced,list(images.values())); rp=reasoning_prompt(model.processor,case['problem'])
   conditions={}; target=OPTION_LETTERS.index(case['target_choice'])
   for (name,img),prob in zip(images.items(),scored):
    completion=generate(model,rp,img); ms=margins(prob); choice=parsed(completion)
    conditions[name]={"option_probabilities":dict(zip(OPTION_LETTERS,prob)),"target_margin":ms[target],"forced_choice":OPTION_LETTERS[int(np.argmax(prob))],"reasoning_completion":completion,"reasoning_choice":choice,"reasoning_correct":choice==case['target_choice']}
   clean=conditions['clean']; ref=conditions['reference_deleted']; rand=[conditions[f'random_deleted_{i}']['target_margin'] for i in range(len(randoms))]
   row={"panel_index":case['panel_index'],"target_choice":case['target_choice'],"reference_boxes_normalized":case['external_annotation']['boxes'],"reference_mask_area_fraction":float(mask.mean()),"reference_neighbor_iou":iou(mask,neighbor),"conditions":conditions,"clean_reasoning_correct":clean['reasoning_correct'],"clean_reasoning_forced_agreement":clean['reasoning_choice']==clean['forced_choice'],"reference_target_margin_drop":clean['target_margin']-ref['target_margin'],"neighbor_target_margin_drop":clean['target_margin']-conditions['neighbor_deleted']['target_margin'],"random_mean_target_margin_drop":clean['target_margin']-float(np.mean(rand)),"reference_minus_random_extra_drop":float(np.mean(rand))-ref['target_margin']}
   rows.append(row); f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n'); f.flush(); print(json.dumps({"done":len(rows),"total":len(cases),"panel_index":case['panel_index']}),flush=True)
 primary=[x for x in rows if x['clean_reasoning_correct'] and x['clean_reasoning_forced_agreement']]
 def mean(k,seq): return float(np.mean([x[k] for x in seq])) if seq else None
 metrics={"schema_version":1,"status":"completed","created_at":datetime.now(timezone.utc).isoformat(),"model":str(a.model.resolve()),"panel":str(a.panel.resolve()),"panel_sha256":sha256_file(a.panel),"slice":{"start":a.start,"stop":a.stop},"case_count":len(rows),"clean_correct_count":sum(x['clean_reasoning_correct'] for x in rows),"primary_clean_correct_and_score_agreement_count":len(primary),"all_cases":{"mean_reference_margin_drop":mean('reference_target_margin_drop',rows),"mean_neighbor_margin_drop":mean('neighbor_target_margin_drop',rows),"mean_random_margin_drop":mean('random_mean_target_margin_drop',rows),"mean_reference_minus_random_extra_drop":mean('reference_minus_random_extra_drop',rows)},"primary_cases":{"mean_reference_margin_drop":mean('reference_target_margin_drop',primary),"mean_neighbor_margin_drop":mean('neighbor_target_margin_drop',primary),"mean_random_margin_drop":mean('random_mean_target_margin_drop',primary),"mean_reference_minus_random_extra_drop":mean('reference_minus_random_extra_drop',primary)},"deletion":"gaussian blur replacement; exact predefined mask; neighbor and 3 random shape/area-matched translations","normal_generation_max_new_tokens":1024,"claim_boundary":"external boxes are diagnostically relevant reference regions; causal contribution requires reference deletion to exceed matched controls"}
 (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n'); print(json.dumps(metrics,sort_keys=True))
if __name__=='__main__': main()
