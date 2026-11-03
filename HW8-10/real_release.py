"""Recheck saved HW6-7 raw outputs and quality gate; prepare existing baseline.

Baseline-versus-itself verifies the already-used fallback; it is not a new
candidate promotion. The original candidate is separately evaluated and rejected.
"""
import json
import os
import shutil
from pathlib import Path
import sys
import release

HERE=Path(__file__).resolve().parent
HW=HERE.parent/'HW6-7'
sys.path.insert(0,str(HW))
from evaluation.contract import fingerprint, read_rows
from evaluation.solution import score, quality_gate
from evaluation.cli import inputs

def read(path): return json.loads(path.read_text())
def main():
    from transformers import AutoTokenizer
    os.chdir(HW)
    cfg, rows, behavioral, baseline_spec, proof, protocol_hash=inputs()
    reports={}
    refs={}
    for role in ('baseline','candidate'):
        rp=HW/'docs'/'reports'/f'{role}-eval.json'; op=HW/'docs'/'reports'/f'{role}-outputs.json'
        report=read(rp); outputs=read(op)
        if report['protocol_sha256']!=protocol_hash or report['golden_sha256']!=fingerprint(rows) or report['baseline_spec_sha256']!=fingerprint(baseline_spec): raise ValueError('evaluation inputs changed')
        if fingerprint(outputs)!=report['outputs_sha256']: raise ValueError('raw outputs changed')
        by_id={item['id']:item for item in outputs}
        if len(by_id)!=len(outputs) or set(by_id)!={r['id'] for r in rows+behavioral}: raise ValueError('raw output IDs changed')
        scored=score(rows,[by_id[r['id']] for r in rows]); bscore=score(behavioral,[by_id[r['id']] for r in behavioral])
        if any(report[k]!=scored[k] for k in ('overall','slices','per_item')) or report['behavioral']!=bscore: raise ValueError('report differs from raw outputs')
        reports[role]=report
        refs[role]={'report':str(rp),'report_sha256':release.file_digest(rp),'outputs':str(op),'outputs_sha256':release.file_digest(op)}
    if reports['baseline']['model']!=baseline_spec: raise ValueError('baseline spec mismatch')
    if reports['candidate']['model']!=read(HW/'docs/reports/candidate-model.json'): raise ValueError('candidate spec mismatch')
    try: quality_gate(reports['baseline'],reports['candidate'],cfg['critical_slices']); candidate={'passed':True}
    except ValueError as exc: candidate={'passed':False,'reason':str(exc)}
    baseline_gate=quality_gate(reports['baseline'],reports['baseline'],cfg['critical_slices'])
    revision=baseline_spec['revision']
    snapshot=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots'/revision
    files=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']
    hashes={name:release.file_digest(snapshot/name) for name in files}
    tok=AutoTokenizer.from_pretrained(baseline_spec['name'],revision=revision,local_files_only=True)
    runtime={'base_revision':revision,'tokenizer_sha256':hashes['tokenizer.json'],
             'template_sha256':release.digest(tok.chat_template.encode()),'dtype':'bfloat16',
             'generation':{'system':cfg['system'],'max_new_tokens':cfg['max_new_tokens'],
                           'enable_thinking':cfg['enable_thinking'],'do_sample':False,'seed':cfg['seed']}}
    manifest={'schema_version':1,'model_id':f"baseline-{revision}",'artifact_uri':baseline_spec['name']+'@'+revision,
              'artifact_root':str(snapshot),'model':baseline_spec,'file_hashes':hashes,
              'model_sha256':release.digest(release.canonical(hashes)), 'protocol_sha256':protocol_hash,
              'golden_sha256':fingerprint(rows),'runtime':runtime,'runtime_sha256':release.digest(release.canonical(runtime))}
    report={k:manifest[k] for k in ('model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256')}
    report.update(status='complete',passed=baseline_gate['passed'],evaluation_run_id=reports['baseline']['evaluation_run_id'],metrics={k:reports['baseline']['overall'][k] for k in ('accuracy','format_rate')},source_reports=refs,decision=baseline_gate,evaluation_role='baseline',scope='baseline compared with itself as existing fallback')
    root=HERE/'.real'; root.mkdir(exist_ok=True)
    shutil.copyfile(HW/cfg['golden'],root/'input.txt')
    prep_config={'tokenizer_revision':revision,'tokenizer_sha256':runtime['tokenizer_sha256'],'template_sha256':runtime['template_sha256'],'max_length':2048,'code_revision':'prepare-count-v1'}
    (root/'config.json').write_text(json.dumps(prep_config,ensure_ascii=False,indent=2)+'\n')
    for name,obj in [('manifest-baseline.json',manifest),('release-report.json',report),('candidate-rejection.json',{'candidate':reports['candidate']['model'],'decision':candidate,'source_reports':refs,'protocol_sha256':protocol_hash})]:
        (root/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'candidate_gate':candidate,'baseline_gate':baseline_gate,'root':str(root),'model_sha256':manifest['model_sha256'],'runtime_sha256':manifest['runtime_sha256']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
