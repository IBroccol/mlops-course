# Связь работы с лекциями и ДЗ1–5

Изучены локальные lecture1–lecture7, задания из учебных архивов, исходники
и отчёты уже выполненных HW1–HW5 общего репозитория.

| Основа | Что требует курс | Что продолжено в ДЗ6+7 |
|---|---|---|
| 1: окружение и инференс | uv.lock, конфиг вместо хардкода, seed, корректные замеры | Python 3.12 и точный lock; модель/бюджет в YAML; greedy из одного протокола |
| 2: анатомия и LoRA | q/k/v/o, gate/up/down, tied embeddings, правильная метрика памяти | sweep варьирует q/v и все 7 проекций; словарь не сохраняется; MPS driver memory |
| 3: данные | собственная тема, CC0-снимок, чистка/группы, DVC, zero leakage | Wikidata v2, полный train-index 2 872; golden по отдельным test-сущностям; DVC и журнал разметки |
| 4: препроцессинг | tokenizer/chat template, labels=-100, маска паддинга, контроль обрезки | используются собственные train.pt и val.pt; повторный encoder сверил все тензоры |
| 5: обучение | LoRA alpha/r, валидация с шага 0, seed, переносимый адаптер | alpha/r=2; token-weighted val на 0/2/4; tokenizer/manifest в пакете |
| 6: эксперименты | отдельные runs, provenance, backend/artifact store, loss gate, registry | 6 finished runs, один бюджет, source snapshot, два volume, candidate и isolated reload |
| 7: качество | отдельный golden, strict format/accuracy, counts/slices, behavioral, fail closed | 60 golden, три критических среза, 3 behavioral, raw+diff, отрицательный тест и отказ champion |

Ключевой результат лекции 7 проявился в реальном прогоне: loss стал лучше,
а accuracy генераций хуже. Поэтому достаточным для выпуска считается
качество на закреплённом протоколе, а не низкий teacher-forced loss.
Медицинские вопросы стартового проекта заменены географическими; тема
предыдущих домашних работ и собственные входные тензоры сохранены.
