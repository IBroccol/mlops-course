"""Строгая оценка JSON-меток и допуск без регрессий. Не требует MLflow/torch."""
import json
import math
from collections import Counter


def _unique(items):
    if not isinstance(items, list) or not items:
        raise ValueError('Нужен непустой список')
    try:
        ids = [item['id'] for item in items]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('ID должны быть уникальными непустыми строками')
        return {item['id']: item for item in items}
    except (KeyError, TypeError) as exc:
        raise ValueError('Некорректный контракт id') from exc


def _counts(items):
    n = len(items)
    correct = sum(i['correct'] for i in items)
    valid = sum(i['valid'] for i in items)
    errors = sum(i['error'] is not None for i in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n, format_rate=valid / n)


def _slices(items):
    return {s: _counts([i for i in items if s in i['slices']])
            for s in sorted({s for i in items for s in i['slices']})}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повторный ключ JSON')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('Не JSON: ' + value)


def score(rows, outputs):
    """Ошибки и невалидные ответы остаются в знаменателе; порядок задаёт rows."""
    questions, answers = _unique(rows), _unique(outputs)
    if questions.keys() != answers.keys():
        raise ValueError('ID вопросов и ответов не совпадают')
    per_item = []
    for row in rows:
        if (not isinstance(row.get('choices'), list) or not row['choices']
                or any(not isinstance(c, str) for c in row['choices'])
                or row.get('expected') not in row['choices']
                or not isinstance(row.get('slices'), list) or not row['slices']
                or any(not isinstance(s, str) or not s for s in row['slices'])
                or len(set(row['slices'])) != len(row['slices'])):
            raise ValueError('Некорректные choices/expected/slices')
        output = answers[row['id']]
        raw, error = output.get('raw', ''), output.get('error')
        if not isinstance(raw, str):
            raw = ''
        if error is not None and not isinstance(error, str):
            raise ValueError('error должен быть строкой или null')
        parsed, valid = None, False
        if error is None:
            try:
                obj = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
                if (isinstance(obj, dict) and set(obj) == {'answer'}
                        and isinstance(obj['answer'], str) and obj['answer'] in row['choices']):
                    parsed, valid = obj['answer'], True
            except (ValueError, RecursionError):
                pass
        per_item.append(dict(id=row['id'], raw=raw, parsed=parsed, valid=valid,
                             correct=valid and parsed == row['expected'], error=error,
                             slices=list(row['slices'])))
    return dict(overall=_counts(per_item), slices=_slices(per_item), per_item=per_item)


def _validate_metrics(actual, expected, context):
    if not isinstance(actual, dict):
        raise ValueError(f'{context}: нет метрик')
    for key in ('n', 'correct', 'valid', 'errors'):
        if type(actual.get(key)) is not int or actual[key] != expected[key]:
            raise ValueError(f'{context}: неверный count {key}')
    for key in ('accuracy', 'format_rate'):
        value = actual.get(key)
        if (type(value) not in (float, int) or not math.isfinite(value)
                or not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'{context}: неверная метрика {key}')


def _validate_score(report, minimum, context):
    if not isinstance(report, dict):
        raise ValueError(f'{context}: нет отчёта')
    items = list(_unique(report.get('per_item')).values())
    if len(items) < minimum:
        raise ValueError(f'{context}: нужно минимум {minimum} примеров')
    for item in items:
        if (type(item.get('valid')) is not bool or type(item.get('correct')) is not bool
                or (item['correct'] and not item['valid']) or 'error' not in item
                or (item['error'] is not None and not isinstance(item['error'], str))
                or (item['error'] is not None and item['valid'])
                or not isinstance(item.get('slices'), list) or not item['slices']
                or any(not isinstance(s, str) or not s for s in item['slices'])
                or len(set(item['slices'])) != len(item['slices'])):
            raise ValueError(f'{context}: некорректный per_item')
    _validate_metrics(report.get('overall'), _counts(items), context)
    expected = _slices(items)
    if not isinstance(report.get('slices'), dict) or report['slices'].keys() != expected.keys():
        raise ValueError(f'{context}: неверный состав срезов')
    for name, metrics in expected.items():
        _validate_metrics(report['slices'][name], metrics, f'{context}/{name}')
    if report['overall']['errors']:
        raise ValueError(f'{context}: ошибки генерации')
    return {i['id']: i for i in items}


def _compatible(a, b, context):
    if a.keys() != b.keys() or any(set(a[i]['slices']) != set(b[i]['slices']) for i in a):
        raise ValueError(f'{context}: разные id/состав срезов')


def quality_gate(baseline, candidate, critical_slices):
    """Отказ на неполных/несовместимых измерениях и на любой заданной регрессии."""
    try:
        for report in (baseline, candidate):
            if (not isinstance(report, dict) or report.get('status') != 'complete'
                    or not isinstance(report.get('model'), dict) or not report['model']):
                raise ValueError('Неполная оценка или отсутствует модель')
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if (not isinstance(baseline.get(key), str) or not baseline[key]
                    or baseline[key] != candidate.get(key)):
                raise ValueError('Несовместимость: ' + key)
        a = _validate_score(baseline, 50, 'baseline')
        b = _validate_score(candidate, 50, 'candidate')
        _compatible(a, b, 'golden')
        if len(baseline['slices']) < 3:
            raise ValueError('Нужны минимум три среза')
        if (not isinstance(critical_slices, list) or not critical_slices
                or any(not isinstance(s, str) for s in critical_slices)
                or not set(critical_slices) <= set(baseline['slices'])):
            raise ValueError('Нет критического среза')
        ba = _validate_score(baseline.get('behavioral'), 3, 'baseline behavioral')
        bb = _validate_score(candidate.get('behavioral'), 3, 'candidate behavioral')
        _compatible(ba, bb, 'behavioral')
        reasons = []
        for key in ('accuracy', 'format_rate'):
            if candidate['overall'][key] < baseline['overall'][key]:
                reasons.append(f'{key}: {candidate["overall"][key]:.4f} < {baseline["overall"][key]:.4f}')
        for name in critical_slices:
            if candidate['slices'][name]['accuracy'] < baseline['slices'][name]['accuracy']:
                reasons.append(f'Критический срез {name}: регрессия accuracy')
        failed = [i for i, item in bb.items() if not item['correct']]
        if failed:
            reasons.append('Не пройдены behavioral: ' + ', '.join(failed))
        if reasons:
            raise ValueError('; '.join(reasons))
        return dict(passed=True, n=len(a), critical_slices=critical_slices,
                    behavioral_passed=len(bb),
                    accuracy_delta=candidate['overall']['accuracy'] - baseline['overall']['accuracy'],
                    format_rate_delta=candidate['overall']['format_rate'] - baseline['overall']['format_rate'],
                    protocol_sha256=candidate['protocol_sha256'])
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Неполный или некорректный отчёт') from exc
