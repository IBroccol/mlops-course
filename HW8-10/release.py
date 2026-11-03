"""Fail-closed release metadata and atomic pointer operations."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(data): return hashlib.sha256(data).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def cache_key(data_path: Path, config: dict) -> str:
    return digest(canonical({'data_sha256': digest(Path(data_path).read_bytes()), 'config': config}))

def validate_release(manifest: dict, report: dict) -> None:
    try:
        if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
            raise ValueError('bad schema')
        for k in ('model_id', 'artifact_uri'):
            if not isinstance(manifest[k], str) or not manifest[k].strip(): raise ValueError(k)
        for k in ('model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256'):
            if not isinstance(manifest[k], str) or not re.fullmatch(r'[0-9a-f]{64}', manifest[k]): raise ValueError(k)
        if not isinstance(manifest['runtime'], dict) or not manifest['runtime']: raise ValueError('runtime')
        if digest(canonical(manifest['runtime'])) != manifest['runtime_sha256']: raise ValueError('runtime mismatch')
        if report['status'] != 'complete' or report['passed'] is not True: raise ValueError('gate rejected')
        if not isinstance(report['evaluation_run_id'], str) or not report['evaluation_run_id'].strip(): raise ValueError('evaluation run')
        for k in ('model_id', 'model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256'):
            if report[k] != manifest[k]: raise ValueError('report mismatch: '+k)
        for k in ('accuracy','format_rate'):
            value=report['metrics'][k]
            if type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1: raise ValueError('metric: '+k)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('missing or invalid release metadata') from exc

def verify_files(manifest, root):
    files = manifest.get('file_hashes')
    if not isinstance(files, dict) or not files: raise ValueError('missing file hashes')
    root = Path(root).resolve()
    actual = {}
    for name, expected in files.items():
        path = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file(): raise ValueError('invalid artifact path')
        actual[name] = file_digest(path)
        if actual[name] != expected: raise ValueError('artifact changed: '+name)
    if digest(canonical(actual)) != manifest['model_sha256']: raise ValueError('model hash mismatch')

def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, suffix='.tmp', delete=False) as f:
            temp = f.name
            json.dump(manifest, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, target); temp = None
    finally:
        if temp is not None: os.unlink(temp)


def verify_evidence(report, manifest):
    """Fixtures are marked; actual releases bind saved reports and raw outputs."""
    if report.get('synthetic') is True:
        return
    refs = report.get('source_reports')
    if not isinstance(refs, dict) or set(refs) != {'baseline', 'candidate'}:
        raise ValueError('missing evaluation evidence')
    for ref in refs.values():
        for name in ('report', 'outputs'):
            if file_digest(Path(ref[name])) != ref[name+'_sha256']:
                raise ValueError('evaluation evidence changed: '+name)
    import sys
    hw = Path(__file__).resolve().parent.parent/'HW6-7'
    sys.path.insert(0, str(hw))
    from evaluation.solution import quality_gate
    baseline = json.loads(Path(refs['baseline']['report']).read_text())
    role = report.get('evaluation_role')
    if role not in ('baseline', 'candidate'): raise ValueError('missing evaluation role')
    selected = json.loads(Path(refs[role]['report']).read_text())
    decision = quality_gate(baseline, selected, report['decision']['critical_slices'])
    if (decision != report['decision'] or selected['model'] != manifest.get('model')
            or selected['evaluation_run_id'] != report['evaluation_run_id']
            or selected['protocol_sha256'] != report['protocol_sha256']
            or selected['golden_sha256'] != report['golden_sha256']
            or any(selected['overall'][k] != report['metrics'][k] for k in ('accuracy','format_rate'))):
        raise ValueError('release decision does not match original baseline evaluation')
