"""One local Prefect flow; reads the existing training result, never trains."""
import argparse
from pathlib import Path
from prefect import flow, task
import lab
import release

@task(name='prepare-with-persistent-content-cache')
def prepare(root):
    # cache lives under the workspace and survives a new Prefect/OS process
    return lab.prepare(Path(root))

@task(name='obtain-and-verify-existing-artifact')
def artifact(root, manifest_name):
    root=Path(root); m=lab.read(root/manifest_name)
    release.verify_files(m, Path(m.get('artifact_root', str(root))))
    return {'manifest_uri':str(root/manifest_name), 'model_id':m['model_id'], 'model_sha256':m['model_sha256']}

@task(name='evaluate-release-contract')
def gate(root, artifact_ref, report_name):
    root=Path(root); m=lab.read(artifact_ref['manifest_uri']); r=lab.read(root/report_name)
    release.validate_release(m,r)
    release.verify_evidence(r,m)
    return {'report_uri':str(root/report_name),'evaluation_run_id':r['evaluation_run_id']}

@task(name='publish-idempotently')
def publish(root, artifact_ref, report_ref, prepared, fail_after_pointer):
    root=Path(root); lab.publish_release(root,lab.read(artifact_ref['manifest_uri']),lab.read(report_ref['report_uri']),prepared,fail_after_pointer)

@flow(name='model-release')
def model_release(root, manifest_name='manifest-B.json', report_name='report-B.json', fail_after_prepare=False, fail_after_pointer=False):
    import uuid
    attempt_id=str(uuid.uuid4())
    try:
        prepared=prepare(root)
        if fail_after_prepare: raise RuntimeError('Инъекция сбоя после подготовки')
        ref=artifact(root, manifest_name); decision=gate(root, ref, report_name)
        publish(root,ref,decision,prepared,fail_after_pointer)
    except Exception as exc:
        lab.append_attempt(Path(root),attempt_id,'failed',str(exc)); raise
    lab.append_attempt(Path(root),attempt_id,'completed')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--manifest',default='manifest-B.json'); p.add_argument('--report',default='report-B.json'); p.add_argument('--fail-after-prepare',action='store_true'); p.add_argument('--fail-after-pointer',action='store_true'); a=p.parse_args()
    model_release(str(a.workspace.resolve()),a.manifest,a.report,a.fail_after_prepare,a.fail_after_pointer)
