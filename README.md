# mlops-course

Сквозной репозиторий YABD-26.

```
HW1/   — окружение, инференс, бенчмарк
HW2/   — анатомия модели, память, LoRA
HW3/   — географический датасет Wikidata, качество, split, DVC
```

```bash
cd HW1 && uv sync && make check
cd HW2 && uv sync && make inspect && make check
cd HW3 && uv sync && make repro && make check
```
