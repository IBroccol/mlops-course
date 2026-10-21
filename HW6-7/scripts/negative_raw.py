"""Независимое доказательство отказа: портим только raw, без подмены counts."""
import copy

from evaluation.cli import current_reports
from evaluation.contract import read_json, read_rows, write_json
from evaluation.solution import score, quality_gate


def main():
    baseline, candidate = current_reports()
    config = read_json('evaluation/params.json')
    rows = read_rows(config['golden'])
    broken_outputs = [dict(id=r['id'], raw='INVALID', error=None) for r in rows]
    broken = copy.deepcopy(candidate)
    broken.update(score(rows, broken_outputs))
    try:
        quality_gate(baseline, broken, config['critical_slices'])
    except ValueError as exc:
        write_json('metrics/negative-raw-gate.json', dict(rejected=True, reason=str(exc),
                   overall=broken['overall'], behavioral_unchanged=True,
                   note='Только копия raw golden; counts пересчитаны scorer, weights/alias не менялись'))
        write_json('metrics/negative-raw-outputs.json', broken_outputs)
        print('PASS: полностью согласованные отчёты с испорченными raw отклонены:', exc)
        return
    raise ValueError('Испорченные raw прошли gate')


if __name__ == '__main__': main()
