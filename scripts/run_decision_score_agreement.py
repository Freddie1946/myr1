#!/usr/bin/env python3
"""Validate that forced A/B/C/D scores represent actual model decisions."""
from __future__ import annotations

import argparse, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

LETTERS="ABCD"
REASONING=("{question}\nGive concise image-grounded reasoning inside <think>...</think>. "
           "Then output exactly one option letter (A, B, C, or D) inside <answer>...</answer>.")
FORCED="{question}\nReturn only one option letter: A, B, C, or D. Do not provide reasoning or other text."

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def prompt(processor,text):
 return processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}],tokenize=False,add_generation_prompt=True)
def parsed(text):
 m=re.findall(r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)",text,re.I|re.S); candidate=m[-1] if m else text
 x=re.search(r"(?:^|\s|\()([ABCD])(?:\)|[\s.,:;\-]|$)",candidate,re.I); return x.group(1).upper() if x else None

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--panel',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--limit',type=int); a=ap.parse_args()
 if a.output_dir.exists(): raise FileExistsError(a.output_dir)
 data=json.loads(a.panel.read_text()); cases=data['cases'][:a.limit]
 a.output_dir.mkdir(parents=True); cfg=AutoConfig.from_pretrained(a.model,local_files_only=True)
 if isinstance(getattr(cfg,'text_config',None),dict): delattr(cfg,'text_config')
 processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,use_fast=False); processor.image_processor.max_pixels=65536; processor.image_processor.min_pixels=3136
 ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in LETTERS]
 if any(len(x)!=1 for x in ids): raise ValueError(ids)
 ids=torch.tensor([x[0] for x in ids],device='cuda')
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,config=cfg,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
 model.generation_config.do_sample=False; model.generation_config.temperature=None; model.generation_config.top_p=None; model.generation_config.top_k=None
 rows=[]; out=a.output_dir/'case_results.jsonl'
 with out.open('w') as f:
  for case in cases:
   path=Path(case['image']);
   if sha(path)!=case['image_sha256']: raise ValueError('image hash')
   image=Image.open(path).convert('RGB')
   p_reason=prompt(processor,REASONING.format(question=case['problem'])); x=processor(text=[p_reason],images=[image],return_tensors='pt'); x={k:v.to('cuda') for k,v in x.items()}
   with torch.inference_mode(): generated=model.generate(**x,max_new_tokens=1024,do_sample=False,use_cache=True)
   completion=processor.tokenizer.decode(generated[0,x['input_ids'].shape[1]:],skip_special_tokens=True)
   p_forced=prompt(processor,FORCED.format(question=case['problem'])); y=processor(text=[p_forced],images=[image],return_tensors='pt'); y={k:v.to('cuda') for k,v in y.items()}
   with torch.inference_mode():
    logits=model(**y,use_cache=False).logits[0,-1].index_select(0,ids).float(); probs=torch.softmax(logits,dim=0)
    constrained=model.generate(**y,max_new_tokens=1,do_sample=False,prefix_allowed_tokens_fn=lambda _batch,_input:ids.tolist(),use_cache=True)
    unconstrained=model.generate(**y,max_new_tokens=4,do_sample=False,use_cache=True)
   lc=LETTERS[int(torch.argmax(probs))]; cc=processor.tokenizer.decode(constrained[0,y['input_ids'].shape[1]:],skip_special_tokens=True).strip(); uc=processor.tokenizer.decode(unconstrained[0,y['input_ids'].shape[1]:],skip_special_tokens=True).strip()
   row={"panel_index":case['panel_index'],"target_choice":case['target_choice'],"reasoning_completion":completion,"reasoning_choice":parsed(completion),"forced_logits_probabilities":dict(zip(LETTERS,probs.cpu().tolist())),"forced_logits_choice":lc,"constrained_generation":cc,"constrained_choice":parsed(cc),"unconstrained_forced_generation":uc,"unconstrained_forced_choice":parsed(uc),"reasoning_correct":parsed(completion)==case['target_choice'],"logits_agree_reasoning":lc==parsed(completion),"constrained_agree_logits":parsed(cc)==lc}
   rows.append(row); f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n'); f.flush(); print(json.dumps({"done":len(rows),"total":len(cases),"panel_index":case['panel_index']}),flush=True)
 metrics={"schema_version":1,"status":"completed","created_at":datetime.now(timezone.utc).isoformat(),"model":str(a.model.resolve()),"model_config_sha256":sha(a.model/'config.json'),"panel":str(a.panel.resolve()),"panel_sha256":sha(a.panel),"count":len(rows),"reasoning_accuracy":sum(x['reasoning_correct'] for x in rows)/len(rows),"forced_logits_vs_reasoning_agreement":sum(x['logits_agree_reasoning'] for x in rows)/len(rows),"constrained_generation_vs_logits_agreement":sum(x['constrained_agree_logits'] for x in rows)/len(rows),"clean_correct_count":sum(x['reasoning_correct'] for x in rows),"clean_correct_logits_agreement":sum(x['reasoning_correct'] and x['logits_agree_reasoning'] for x in rows)/max(1,sum(x['reasoning_correct'] for x in rows)),"normal_generation_max_new_tokens":1024,"interpretation":"A-D are single tokens; forced logits equal candidate one-token sequence likelihood under the forced-choice interface. Agreement with normal reasoning generation determines whether this score may be used as a decision endpoint."}
 (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n'); print(json.dumps(metrics,sort_keys=True))
if __name__=='__main__': main()
