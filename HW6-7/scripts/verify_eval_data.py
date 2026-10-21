"""Проверка приложения с данными без базовой модели, сервера и тензоров train."""
import hashlib
import json
from pathlib import Path
from collections import Counter

from evaluation.contract import read_json, read_rows, validate_golden


def main():
    root = Path('data/eval')
    golden, train, behavior = [read_rows(root/f'{name}.jsonl') for name in ('golden','training-index','behavioral')]
    proof = validate_golden(golden, train, behavior)
    source = {r['entity_id']: r for r in read_rows_source(root/'source-records.jsonl')}
    for row in golden:
        record = source[row['group_id'].split('/')[1]]
        if hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() != row['source']['record_sha256']:
            raise ValueError('Изменена карточка источника: ' + row['id'])
        review = row['review']
        if any(record[k] != v for k,v in review['source_fact'].items()):
            raise ValueError('Факт не совпал с исходником')
        if review['options'][row['expected']] != review['answer_text']:
            raise ValueError('Метка не совпала с эталоном')
    prior = read_json(root/'provenance.json')
    if proof['n_train_index'] != prior['tokenized_examples_exactly_reencoded']:
        raise ValueError('Неполный train-index')
    print(json.dumps(dict(**proof, labels=dict(Counter(r['expected'] for r in golden)),
                         slices=dict(Counter(s for r in golden for s in r['slices'])),
                         source_records_checked=len(golden)), ensure_ascii=False, indent=2))


def read_rows_source(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


if __name__ == '__main__': main()
