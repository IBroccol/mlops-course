"""Measure full HTTP requests after warmup; retain successes and failures."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

URL='http://127.0.0.1:8091'

def request(path, body=None, timeout=180):
    payload=None if body is None else json.dumps(body,ensure_ascii=False).encode()
    req=urllib.request.Request(URL+path,data=payload,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r: return r.status,json.load(r)
    except urllib.error.HTTPError as e: return e.code,json.load(e)

def main():
    end=time.monotonic()+180
    while time.monotonic()<end:
        try:
            status,body=request('/ready',timeout=3)
            if status==200: break
        except Exception: pass
        time.sleep(1)
    else: raise RuntimeError('model not ready within 180s')
    prompt='Карточка Wikidata: {"name":"Париж","country":"Франция"}. В какой стране находится город? Варианты: A: Франция; B: Германия; C: Италия.'
    bad=request('/generate',{'prompt':prompt,'max_new_tokens':999})
    smoke=request('/generate',{'prompt':prompt,'max_new_tokens':32})
    Path('evidence/real-smoke.json').write_text(json.dumps({'ready':body,'invalid':bad,'response':smoke},ensure_ascii=False,indent=2)+'\n')
    if smoke[0]!=200: raise RuntimeError('smoke generation failed')
    records=[]; summaries=[]
    for concurrency in (1,2):
        def one(index):
            client_id=str(uuid.uuid4()); started=time.perf_counter(); status=None; data={}; error=None
            try:
                status,data=request('/generate',{'prompt':prompt,'max_new_tokens':32})
                if status!=200: error=str(data)
            except Exception as exc: error=f'{type(exc).__name__}: {exc}'
            return {'client_id':client_id,'index':index,'concurrency':concurrency,'status':status,'error':error,'latency_seconds':time.perf_counter()-started,'request_id':data.get('request_id'),'generation_seconds':data.get('generation_seconds'),'model_id':data.get('model_id'),'output_tokens':data.get('output_tokens')}
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex: batch=list(ex.map(one,range(8)))
        duration=time.perf_counter()-start; records.extend(batch)
        good=[x for x in batch if x['status']==200 and not x['error']]; lat=[x['latency_seconds'] for x in good]
        summaries.append({'concurrency':concurrency,'n':len(batch),'successes':len(good),'errors':len(batch)-len(good),'batch_seconds':duration,'successful_rps':len(good)/duration,'mean_success_latency':sum(lat)/len(lat) if lat else None,'max_success_latency':max(lat) if lat else None})
    Path('evidence/benchmark.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    Path('evidence/benchmark-summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
