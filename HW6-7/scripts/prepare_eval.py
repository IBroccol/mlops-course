"""Географическая регрессия по зафиксированному снимку, без запросов в сеть."""
import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from evaluation.contract import validate_golden, write_json


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, data):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in data))


def main():
    root = Path('data/eval'); root.mkdir(parents=True, exist_ok=True)
    hw3, hw4 = Path('../HW3'), Path('../HW4')
    train = rows(hw3 / 'data/train.jsonl')
    if sha(hw3 / 'data/train.jsonl') != sha(hw4 / 'data/train.jsonl'):
        raise ValueError('Исходный train ДЗ3 и вход токенизации ДЗ4 различаются')
    tokenized = torch.load(hw4 / 'data/tokenized/train.pt', weights_only=True)
    if [r['id'] for r in train] != [r['id'] for r in tokenized['examples']]:
        raise ValueError('Полный train-index не совпал с тензорами ДЗ4')
    # Повторно вызываем именно encoder ДЗ4, а не приблизительное сравнение текста.
    cfg4 = yaml.safe_load((hw4 / 'params.yaml').read_text())
    p = yaml.safe_load(Path('params.yaml').read_text())
    tok = AutoTokenizer.from_pretrained(p['model']['name'], revision=p['model']['revision'], local_files_only=True)
    saved_modules = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
    for k in saved_modules: del sys.modules[k]
    sys.path.insert(0, str(hw4.resolve()))
    try:
        from src.tokenize_data import encode_example
        for record, example in zip(train, tokenized['examples']):
            encoded = encode_example(tok, record, cfg4, cfg4['tokenize']['max_seq_len'])
            encoded.pop('_meta')
            if encoded != example:
                raise ValueError('Токенизация/маска не совпала: ' + record['id'])
    finally:
        sys.path.pop(0)
        for k in list(sys.modules):
            if k == 'src' or k.startswith('src.'): del sys.modules[k]
        sys.modules.update(saved_modules)
    training = [dict(id=r['id'], group_id=r['topic'], input=r['messages'][1]['content']) for r in train]
    heldout = {r['topic'] for r in rows(hw3 / 'data/test.jsonl')}
    sources = sorted((hw3 / 'sources/data').glob('world_*.jsonl'))
    facts = {r['entity_id']: r for path in sources for r in rows(path)}
    cities = sorted((r for r in facts.values() if 'city/' + r['entity_id'] in heldout), key=lambda r:r['entity_id'])
    countries = sorted((r for r in facts.values() if 'country/' + r['entity_id'] in heldout), key=lambda r:r['entity_id'])
    golden, reviews = [], []
    all_countries = sorted({r['country'] for r in facts.values() if r['kind'] == 'city'})
    all_capitals = sorted({r['capital'] for r in facts.values() if r['kind'] == 'country'})
    rng = random.Random(67)
    def add(record, task, question, answer, pool):
        number = len(golden)
        options = rng.sample([x for x in pool if x != answer], 2)
        at = number % 3
        options.insert(at, answer)
        labels = ['A', 'B', 'C']
        item_id = 'golden-' + record['entity_id'] + '-' + task
        # Минимальная карточка исключает лишние факты; эталон модели не передаётся.
        keys = {'country_lookup': ['name','country'], 'hemisphere': ['name','latitude','longitude'],
                'capital_lookup': ['name','capital']}[task]
        card = {k: record[k] for k in keys}
        content = 'Карточка Wikidata: ' + json.dumps(card, ensure_ascii=False) + '\n' + question
        content += '\n' + '\n'.join(f'{label}: {option}' for label, option in zip(labels, options))
        content += '\nchoices: A, B, C. Верни только JSON с полем answer.'
        row = dict(id=item_id, group_id=record['kind'] + '/' + record['entity_id'], input=content,
                   expected=labels[at], choices=labels, slices=[task, record['kind']],
                   source={'url': record['source_url'], 'snapshot_date': '2026-09-17',
                           'record_sha256': hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()},
                   review={'method': 'source-card review by Codex, v1', 'answer_text': answer,
                           'source_fact': card, 'options': dict(zip(labels, options))})
        assert options[at] == answer and len(set(options)) == 3
        golden.append(row)
        reviews.append([item_id,record['name'],task,labels[at],answer,json.dumps(card,ensure_ascii=False)])
    for r in cities[:20]:
        add(r, 'country_lookup', 'В какой стране находится город согласно карточке?', r['country'], all_countries)
    hemispheres = ['северное и восточное', 'северное и западное', 'южное и восточное', 'южное и западное']
    for r in cities[20:40]:
        if not r['latitude'] or not r['longitude']: raise ValueError('Неоднозначные полушария на оси')
        answer = ('северное' if r['latitude'] > 0 else 'южное') + ' и ' + ('восточное' if r['longitude'] > 0 else 'западное')
        add(r, 'hemisphere', 'В каких полушариях лежит объект по знакам широты и долготы?', answer, hemispheres)
    for r in countries[:20]:
        add(r, 'capital_lookup', 'Какая столица записана в карточке страны?', r['capital'], all_capitals)
    if len(golden) != 60: raise ValueError('Нужны ровно 60 разных сущностей')
    behavior = []
    original = golden[0]
    for kind in ('format', 'paraphrase'):
        row = dict(original, id='behavior-' + kind, type=kind)
        if kind == 'paraphrase':
            row['input'] = row['input'].replace('В какой стране находится город согласно карточке?',
                'Опираясь исключительно на карточку, выбери государство, в котором расположен этот город.')
        behavior.append(row)
    behavior.append(dict(id='behavior-refusal', group_id='outside-geography/software', type='refusal',
                        input='Напиши Python-функцию сортировки списка. choices: REFUSE.', expected='REFUSE',
                        choices=['REFUSE'], slices=['outside_domain'], source='Авторский запрос вне границы географической задачи',
                        review={'method':'Проверено по system-инструкции: программирование вне географии'}))
    proof = validate_golden(golden, training, behavior)
    if {r['group_id'] for r in behavior} & {r['group_id'] for r in training}:
        raise ValueError('Утечка behavioral/train')
    dependencies = [hw3 / 'data/train.jsonl', hw3 / 'data/test.jsonl', hw4 / 'data/train.jsonl',
                    hw4 / 'data/tokenized/train.pt', hw4 / 'data/tokenized/val.pt', hw4 / 'params.yaml',
                    *sources, *sorted((hw4/'src').glob('*.py'))]
    proof.update(source_sha256={str(p):sha(p) for p in dependencies},
                 tokenized_examples_exactly_reencoded=len(train), entity_groups=len({r['group_id'] for r in golden}),
                 train_groups=len({r['group_id'] for r in training}),
                 note='Golden извлечён из test ДЗ3; эти 60 групп теперь regression, не независимый final audit. '
                      'ID, группы, исходные вопросы и все тензоры train проверены; смысловая утечка вручную ограничена review по сущности.')
    write_rows(root/'golden.jsonl', golden); write_rows(root/'training-index.jsonl', training)
    write_rows(root/'behavioral.jsonl', behavior)
    write_json(root/'provenance.json', proof)
    write_rows(root/'source-records.jsonl', [facts[r['group_id'].split('/')[1]] for r in golden])
    with (root/'annotation-review.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t'); writer.writerow(['id','entity','task','label','answer','source_fact']); writer.writerows(reviews)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
