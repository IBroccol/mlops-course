# ДЗ6+7 — эксперименты MLflow и проверка качества

Продолжение общего репозитория ДЗ1–5. Тема: географический справочник по
карточкам Wikidata. Реальные результаты от 21 октября 2026, Apple M3 Pro,
36 ГБ, MPS, Qwen3-0.6B, bf16. Это одна работа к лекциям 6 и 7.

**Итог: candidate сохранён, champion не назначен.** Лучший по loss адаптер
`all-r8-high`: val 1,6095 → 0,8719; на 60 golden-вопросах accuracy
49/60 → 46/60, критический hemisphere 10/20 → 6/20. Формат 59/60 → 60/60,
behavioral 3/3 у обеих версий. Порог после результата не менялся.

- [Единый вывод и пять разобранных пар](docs/conclusion.md).
- [Четыре дефекта и доказательства](docs/defects.md).
- [Шесть runs из MLflow](docs/comparison.csv).
- [Протокол до оценки](docs/protocol.md), [данные и доступ](docs/data-access.md).
- [9/9 + 24 теста, полный фактический лог](docs/reports/check-all.txt).
- [Все отчёты, raw, срезы и отказ](docs/reports).
- Приложения с данными, отчётами и видео: локальный каталог `submission/`.
  Публикация приложений в публичном Release ожидает разрешения владельца.

## Быстрая проверка без модели

```bash
cd HW6-7
uv sync --frozen
make check-eval
# После скачивания приложения с данными:
tar -xzf hw67-evaluation-data.tar.gz
uv run python -m scripts.verify_eval_data
```

В tests/ — выданные 24 теста без изменений. Отпечаток **c42ac7788026**.
Синтетические fixtures не являются измерениями Qwen. Логи unit-тестов могут
печатать `champion -> models:/fixture-model/2`: это изолированный mock,
не изменение реального реестра. Фактическое состояние — registry-state.json.

## Повторение полного маршрута

Нужны Python 3.12, uv, Docker + Compose v2. Входы — собственные тензоры
`../HW4/data/tokenized/{train,val}.pt`; они не входят в Git/архив сдачи.
Для MPS выполняйте вне ограниченного sandbox. `auto` выбирает CUDA/MPS/CPU;
на CPU поменяйте dtype на float32 **до нового sweep**. Базовая ревизия
закреплена в params.yaml, требуется её HF-кеш или доступ к Hugging Face.

```bash
uv sync --frozen
make check-eval
make up
make sweep                   # 6 отдельных runs × 4 шага, около 1–2 минут на этой машине
make report
make candidate               # loss gate → candidate
make baseline-init           # только в новом checkout: существующий baseline не перезаписывать
make evaluate                # две модели, единый prompt/decoding
make eval                    # при регрессии ожидаемо завершится ненулевым кодом
make check-negative
uv run python -m scripts.negative_raw
# make promote               # выполнить только после PASS; в этой сдаче пропущено
make check-storage
make check-all
```

Baseline уже сохранён в Git. Для воспроизведения этой сдачи команда
`baseline-init` не нужна; она предназначена только для первого закрепления.
`make baseline-update` допустим только после принятия champion и сохраняет
историю. Нельзя менять baseline ради принятия неудачного кандидата.

Сервер: [MLflow UI](http://127.0.0.1:5006), compose project `mlops-hw6`.
Postgres хранит метаданные, `artifacts` volume — пакеты/исходники/ответы;
клиент загружает через сервер. `make down` сохраняет volumes. Не удаляйте их
через `down -v`, если нужны сохранённые эксперименты.

Реальные URI: `models:/wikidata-geography-adapter@candidate` и
`models:/wikidata-geography-adapter/1`. Champion отсутствует. Offline-проверка
загружает candidate в новом изолированном процессе без локального params.yaml.
База из закреплённой ревизии должна быть заранее закеширована.

## Данные и подготовка

`dvc.lock` фиксирует 60 golden/3 behavioral/2 872 train-index и их источник.
`uv run dvc pull` работает при доступности локального remote; для другого
компьютера предусмотрено локальное приложение. `uv run dvc repro` требует
исходных файлов ДЗ3–4 и повторно сверяет все input_ids, masks и labels с
encoder ДЗ4. Не запускайте новое приготовление/обучение для просмотра уже
сохранённых результатов. Обновление данных меняет протокол и требует новой
оценки; train/config/src меняют выборку экспериментов и требуют нового sweep.

`docs/reports/` намеренно содержит копии отчётов из игнорируемого metrics/.
`uv run python -m scripts.finalize` экспортирует доказательства и добавляет
их в evaluation run; код предназначен для этой сдачи с отклонённым кандидатом.
Веса, .venv, тензоры, volumes и секреты в Git и приложениях отсутствуют.

Озвученное видео: автоматически подготовленный обзор реальных снимков UI,
измерений и логов, с явной отметкой автоматической озвучки. Исходный текст
озвучки сохранён в docs/video-script.json. Скринкаст от лица человека не
имитируется. [Требования части B](EVALUATION.md) сохранены из архива курса.
