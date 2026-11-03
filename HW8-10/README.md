# ДЗ 8–10: выпуск и API

Решение в этом каталоге. `lab.py` сохраняет совместимый интерфейс стенда из архива; `flow.py` — локальный Prefect-flow, `service.py` — FastAPI со stub и Transformers backend. Исходный проект сохранён коммитом `1d2b216`. Оригинальные 13 тестов не изменялись.

## Быстрый запуск

Python 3.12; для установки зависимостей нужен доступ к PyPI:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
make check
.venv/bin/python lab.py --workspace .lab-demo init
PREFECT_HOME="$PWD/.prefect" PREFECT_SERVER_ANALYTICS_ENABLED=false \
  .venv/bin/python flow.py --workspace .lab-demo
.venv/bin/python service.py --workspace .lab-demo --warmup-seconds 2 --port 8091
```

`make check` содержит 13 исходных тестов и новые тесты; HTTP-проверкам нужны разрешённые локальные сокеты. `evidence/integration-red.txt` и `evidence/atomic-red.txt` — исходный код; `evidence/check-green.txt` — исправленный. В `defects.csv` семь причин. Проверка ошибки диска использует mock, не заполняет диск.

`demo_flow.sh` создаёт новый `.lab-flow-v3`, выполняет независимые процессы, намеренные сбои и отказ. Для повторного запуска выберите новое имя каталога в скрипте; журнал показан в `evidence/prefect-demo.txt`. Кеш хранится в рабочем каталоге, а не в памяти Prefect; ключ включает байты, конфигурацию и фактический хеш кода функции подготовки. Результат подготовки — проверка UTF-8 и непустых строк, число строк и хеш, не обучение. `attempts.jsonl` хранит попытки, `publications.jsonl` — события. Один писатель; атомарная замена указателя не решает конкурентную запись. Повтор после сбоя между заменой и журналом восстанавливает событие.

## Реальная модель и доступ к данным

Результаты ДЗ 6–7 находятся в соседнем `../HW6-7`: `docs/reports/*-eval.json` и `*-outputs.json`, `evaluation/`, `data/eval/`, а также исходный protocol. Golden и остальные данные могут быть восстановлены из приложения ДЗ 6–7 или локального DVC remote; инструкции в `../HW6-7/docs/data-access.md`. Веса в Git не включены. Нужна локальная копия Hugging Face модели `Qwen/Qwen3-0.6B` ревизии `c1899de289a04d12100db370d81485cdf75e47ca`. Контрольные суммы конкретных файлов сохранены в `evidence/manifest-baseline.json`; путь в сохранённом manifest относится к машине автора и пересоздаётся скриптом.

Используйте окружение предыдущей работы `../HW6-7/.venv`, закреплённое в её `uv.lock`, и установленное в текущем окружении Prefect. `real_release.py` повторно проверяет исходные данные, протокол, все сырые ответы и отчёты через scorer и quality gate предыдущей работы. Он создаёт `.real/manifest-baseline.json`, `.real/release-report.json` и `.real/candidate-rejection.json`. Подготовка не обучает модель. Это полная переоценка **сохранённых ответов**, не повторная генерация.

```sh
../HW6-7/.venv/bin/python real_release.py
PREFECT_HOME="$PWD/.prefect" PREFECT_SERVER_ANALYTICS_ENABLED=false \
  .venv/bin/python flow.py --workspace .real --manifest manifest-baseline.json --report release-report.json
../HW6-7/.venv/bin/python service.py --workspace .real --backend transformers --warmup-seconds 0 --port 8091
# В другом терминале:
../HW6-7/.venv/bin/python benchmark.py
```

Сервис запускается на `127.0.0.1`; принимает только оценённый лимит 32 новых токена. Неизвестный лимит/пустой ввод отклоняются до вызова генерации. Модель загружается из файлов, хеши которых проверены; запись указателя не меняет её в памяти. Метаданные отражают загруженный backend. `evidence/real-smoke.json`, `benchmark.jsonl`, `benchmark-summary.json` — реальные результаты. Учебные версии A/B явно помечены STUB ONLY.

Baseline сравнивается с самой собой для проверки уже существующей контрольной версии; это не успешный gate адаптера и не присвоение alias champion. Кандидат в ДЗ 6–7 действительно отклонён. Сохранённые тогда результаты не содержали отдельно подписанных SHA файлов токенизатора и весов: manifest закрепляет текущие локальные файлы указанной revision, а их побайтовое совпадение с файлами во время старой оценки независимо не доказано. Для строгого аудита надо повторить полную генерационную оценку в этом окружении и сохранить исходные SHA на момент оценки.

`submission/hw8910-demo.mp4` — видео 5:20 по сохранённым доказательствам с синтезированным голосом; это не запись живого экрана. Скрипт `make_video.py` собирает его с помощью Pillow, imageio-ffmpeg и macOS `say`. Видео и зависимости для его сборки не входят в Git. Сценарий, ограничения и таблица триггеров — в `REPORT.txt`.
