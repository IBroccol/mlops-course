"""FastAPI serving: loaded manifest is an immutable snapshot until restart."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictInt, StrictStr
import uvicorn
import release


class StubBackend:
    def __init__(self, manifest):
        self.loaded_id=manifest['model_id']; self.runtime_sha256=manifest['runtime_sha256']; self.calls=0
    def generate(self, prompt, limit):
        self.calls+=1; time.sleep(.01)
        return {'answer':f'STUB ONLY: {self.loaded_id}', 'output_tokens':1}

class TransformersBackend:
    def __init__(self, manifest):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        runtime=manifest['runtime']; spec=manifest['model']; self.torch=torch
        self.loaded_id=manifest['model_id']; self.runtime_sha256=manifest['runtime_sha256']; self.calls=0
        self.device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        if spec.get('revision') != runtime['base_revision']: raise ValueError('base revision mismatch')
        model_path=manifest['artifact_root']
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        if release.digest(self.tokenizer.chat_template.encode()) != runtime['template_sha256'] or manifest['file_hashes'].get('tokenizer.json') != runtime['tokenizer_sha256']:
            raise ValueError('tokenizer/template mismatch')
        self.model=AutoModelForCausalLM.from_pretrained(model_path,dtype=getattr(torch,runtime['dtype']),local_files_only=True)
        if spec['kind']=='adapter':
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model, spec['adapter_dir'], is_trainable=False)
        self.model=self.model.to(self.device).eval(); self.generation=runtime['generation']
    def generate(self, prompt, limit):
        self.calls+=1; t=self.torch
        text=self.tokenizer.apply_chat_template([{'role':'system','content':self.generation['system']}, {'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=self.generation['enable_thinking'])
        batch=self.tokenizer(text, return_tensors='pt').to(self.device)
        if batch['input_ids'].shape[1]>2048: raise ValueError('input exceeds 2048 tokens')
        with t.inference_mode():
            out=self.model.generate(**batch,max_new_tokens=limit,do_sample=False,pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        tokens=out[0,batch['input_ids'].shape[1]:]
        return {'answer':self.tokenizer.decode(tokens,skip_special_tokens=True),'output_tokens':len(tokens)}

class GenerateRequest(BaseModel):
    prompt: StrictStr
    max_new_tokens: StrictInt = 32

def create_app(root, warmup_seconds=0, backend_kind='stub'):
    state={'ready':False,'backend':None,'error':None}; lock=threading.Lock()
    def load():
        try:
            manifest=json.loads((root/'accepted.json').read_text())
            release.verify_files(manifest, Path(manifest.get('artifact_root',str(root))))
            if release.digest(release.canonical(manifest['runtime'])) != manifest['runtime_sha256']: raise ValueError('runtime hash mismatch')
            backend=StubBackend(manifest) if backend_kind=='stub' else TransformersBackend(manifest)
            if warmup_seconds: time.sleep(warmup_seconds)
            if backend_kind!='stub': backend.generate('Проверка готовности.', manifest['runtime']['generation']['max_new_tokens'])
            state.update(backend=backend, manifest=manifest, ready=True)
        except Exception as exc: state['error']=str(exc)
    @asynccontextmanager
    async def lifespan(app):
        thread=threading.Thread(target=load,daemon=True); thread.start()
        yield
    app=FastAPI(lifespan=lifespan)
    @app.get('/health')
    def health(): return {'alive':True}
    @app.get('/ready')
    def ready():
        if not state['ready']: raise HTTPException(503, detail=state['error'] or 'loading')
        return {'ready':True,'model_id':state['backend'].loaded_id}
    @app.get('/debug')
    def debug():
        if backend_kind!='stub': raise HTTPException(404)
        b=state['backend']; return {'loaded_id':b.loaded_id if b else None,'calls':b.calls if b else 0,'warmup_finished':state['ready']}
    @app.post('/generate')
    def generate(request: GenerateRequest):
        if not request.prompt.strip() or len(request.prompt)>10000: raise HTTPException(422,detail='invalid prompt')
        if not state['ready']: raise HTTPException(503,detail='model not ready')
        limit=state['manifest']['runtime']['generation']['max_new_tokens']
        if request.max_new_tokens != limit: raise HTTPException(422,detail=f'max_new_tokens must equal evaluated limit {limit}')
        started=time.perf_counter()
        try:
            with lock:
                computation=time.perf_counter(); result=state['backend'].generate(request.prompt,request.max_new_tokens)
                result['generation_seconds']=time.perf_counter()-computation
        except ValueError as exc: raise HTTPException(422,detail=str(exc)) from exc
        b=state['backend']; result.update(model_id=b.loaded_id,runtime_sha256=b.runtime_sha256,request_id=str(uuid.uuid4()),total_seconds=time.perf_counter()-started)
        return result
    return app

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,default=Path('.lab')); p.add_argument('--port',type=int,default=8091); p.add_argument('--warmup-seconds',type=float,default=5); p.add_argument('--backend',choices=['stub','transformers'],default='stub')
    a=p.parse_args(); uvicorn.run(create_app(a.workspace,a.warmup_seconds,a.backend),host='127.0.0.1',port=a.port,log_level='warning')
if __name__=='__main__': main()
