"""Экспорт реальных доказательств; не меняет пороги, исходные ответы и champion."""
import json
import shutil
from pathlib import Path

from mlflow import MlflowClient
from src.common import connect, load_params
from evaluation.contract import read_json, write_json


def main():
    p=load_params(); connect(p); client=MlflowClient()
    base=read_json('metrics/baseline-eval.json'); cand=read_json('metrics/candidate-eval.json')
    spec=cand['model']; version=client.get_model_version_by_alias(spec['name'],'candidate')
    aliases = client.get_registered_model(spec['name']).aliases
    champion_state = None
    if 'champion' in aliases:
        champion = client.get_model_version(spec['name'], aliases['champion'])
        champion_state = dict(version=champion.version, run_id=champion.run_id)
    write_json('metrics/registry-state.json', dict(candidate=spec,champion=champion_state,
               release_decision='rejected', quality_gate=read_json('metrics/quality-gate.json')))
    client.set_model_version_tag(spec['name'],spec['version'],'quality_gate','rejected')
    client.set_model_version_tag(spec['name'],spec['version'],'evaluation_run_id',cand['evaluation_run_id'])
    client.update_model_version(spec['name'],spec['version'],description=(
        'Wikidata geography, snapshot 2026-09-17. LoRA r8/all7, lr=3e-4, 4 steps, seed42. '
        'Teacher-forced val 1.6095 -> 0.8719 (8 items). Golden accuracy 49/60 -> 46/60; '
        'hemisphere 10/20 -> 6/20. Format 59/60 -> 60/60; behavioral 3/3 both. '
        'Loss gate passes, quality gate rejects. Candidate only, no champion. '
        'Review: Codex; training-index has all 2872 rows. Not an independent final audit.'))
    report_dir=Path('docs/reports'); report_dir.mkdir(parents=True,exist_ok=True)
    for path in Path('metrics').glob('*.json'):
        if path.name == 'negative-raw-outputs.json': continue
        shutil.copy2(path,report_dir/path.name)
        client.log_artifact(cand['evaluation_run_id'],str(path),'evidence')
    shutil.copy2('data/eval/provenance.json',report_dir/'data-provenance.json')
    for name in ['check-all.txt','negative-raw.txt','gate.txt','storage.txt']:
        path=report_dir/name
        if path.exists(): client.log_artifact(cand['evaluation_run_id'],str(path),'evidence')
    client.set_tag(cand['evaluation_run_id'],'quality_gate','rejected')
    for role,report in [('baseline',base),('candidate',cand)]:
        for name,metrics in report['slices'].items():
            for key in ('n','correct','valid','errors','accuracy','format_rate'):
                client.log_metric(cand['evaluation_run_id'],f'{role}.slice.{name}.{key}',metrics[key])
    print('Сохранены отчёты и решение об отказе. Champion:', champion_state)


if __name__ == '__main__': main()
