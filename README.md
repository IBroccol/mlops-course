# mlops-course

Сквозной репозиторий YABD-26.

```
HW1/   — окружение, инференс, бенчмарк
HW2/   — анатомия модели, память, LoRA
HW3/   — географический датасет Wikidata, качество, split, DVC
HW4/   — chat template, masking, токенизация и packing
HW5/   — LoRA: all_layers / freeze14, валидация, переносимость адаптера
```

```bash
cd HW1 && uv sync && make check
cd HW2 && uv sync && make inspect && make check
cd HW3 && uv sync && make repro && make check
cd HW4 && uv sync && make tokenize && make check
cd HW5 && uv sync && make train && make compare && make check
```
