# Данные и доступ проверяющему

Оценочные данные версионируются DVC: стадия `prepare_eval` и `dvc.lock`.
Локальный remote по умолчанию: `MLOPS/dvc_remote_hw67` рядом с общим
репозиторием, а не публичный облачный сервис. Путь переносим относительно
`.dvc/config`; удалённому проверяющему он сам по себе недоступен.

К GitHub Release этой работы приложен `hw67-evaluation-data.tar.gz`:
`golden.jsonl`, `behavioral.jsonl`, полный `training-index.jsonl`,
`source-records.jsonl`, `annotation-review.tsv`, `provenance.json`.
В архиве нет весов, тензоров обучения, volumes и окружений.

После скачивания архива из Release в каталог `HW6-7`:

```bash
mkdir -p data/eval
tar -xzf hw67-evaluation-data.tar.gz
uv sync --frozen
uv run python -m scripts.verify_eval_data
```

Если у вас есть доступ к локальному DVC remote автора:

```bash
uv run dvc pull
uv run python -m scripts.verify_eval_data
```

Для воспроизведения подготовки с нуля нужны данные предыдущих ДЗ:
`HW3/data/train.jsonl`, `test.jsonl`, снимки `HW3/sources/data` и
`HW4/data/train.jsonl`, `data/tokenized/{train,val}.pt`.
Они не включены в Git/архив сдачи согласно заданию. Точные SHA256 этих
входов записаны в `data/eval/provenance.json` и в отчёте сдачи.
`dvc repro` повторно проверяет все 2 872 токенизированных примера
encoder'ом ДЗ4, включая input_ids, labels, attention_mask и id.

Полный индекс используется для контроля утечки даже при бюджете train=32.
Ноль пересечений ID/групп/нормализованных вопросов не является доказательством
отсутствия всех смысловых перефразов. Golden выбран по целым test-сущностям;
разметка сверена с локальными карточками агентом Codex, без выдачи этой
проверки за независимую экспертизу человека или проверку текущего Wikidata.
Лицензия фактов Wikidata: CC0, сохранены ссылки на источник и дата снимка.
