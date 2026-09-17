# Конкретный diff пяти дефектов

Ниже приведён unified diff между исходным состоянием пайплайна и исправленной
реализацией. Изменения собственного датасета и форматирование не включены,
чтобы причины и исправления дефектов были видны отдельно.

## 1. Данные попадали в Git и не полностью управлялись DVC

```diff
--- a/.gitignore
+++ b/.gitignore
@@
 dvc_remote/
+data/

--- a/.dvc/config
+++ b/.dvc/config
@@
 [core]
     autostage = true
+    remote = local
+['remote "local"']
+    url = dvc_remote

--- a/dvc.yaml
+++ b/dvc.yaml
@@ split.outs
     outs:
       - data/train.jsonl
+      - data/val.jsonl
+      - data/test.jsonl
```

Эффект: `git ls-files data/` возвращает 0 файлов, а автоматическая проверка видит все пять
артефактов DVC: raw, clean, train, validation и test.

## 2. Смена версии не инвалидировала pipeline

```diff
--- a/dvc.yaml
+++ b/dvc.yaml
@@ collect
   collect:
     cmd: python -m src.collect
     deps:
-      # TODO: добавить сюда путь к СВОЕМУ источнику данных.
+      - sources/data
       - src/collect.py
       - src/config.py
+    params:
+      - collect

@@ clean
   clean:
     cmd: python -m src.clean
     deps:
+      - data/raw.jsonl
       - src/clean.py
```

Эффект: изменение `collect.version` теперь меняет сигнатуру collect, а новый
`raw.jsonl` инвалидирует clean. Проверка получила разные MD5:
`723d291a` для v1 и `39c777bc` для v2.

## 3. Построчный split создавал утечку

```diff
--- a/src/split.py
+++ b/src/split.py
@@
-def row_split(count: int, ratios: dict[str, float], seed: int) -> list[str]:
-    order = list(range(count))
-    random.Random(seed).shuffle(order)
-    labels = [""] * count
-    ...
-    return labels
+def group_split(
+    examples: list[Example], ratios: dict[str, float], seed: int
+) -> dict[str, list[Example]]:
+    grouped: dict[str, list[Example]] = {}
+    for ex in examples:
+        grouped.setdefault(normalize_group(ex.topic), []).append(ex)
+    groups = list(grouped.items())
+    random.Random(seed).shuffle(groups)
+    groups.sort(key=lambda item: len(item[1]), reverse=True)
+    buckets = {name: [] for name in ratios}
+    targets = {name: len(examples) * ratio for name, ratio in ratios.items()}
+    for _, rows in groups:
+        label = max(ratios, key=lambda name: targets[name] - len(buckets[name]))
+        buckets[label].extend(rows)
+    return buckets
@@
-    labels = row_split(len(examples), cfg["ratios"], cfg["seed"])
-    buckets = {name: [] for name in cfg["ratios"]}
-    for label, ex in zip(labels, examples):
-        buckets[label].append(ex)
+    buckets = group_split(examples, cfg["ratios"], cfg["seed"])
@@
     rep = report(...)
+    if any(rep[key] for key in
+           ("id_overlap", "text_overlap", "group_overlap", "near_dup_pairs")):
+        raise RuntimeError(f"split создал контаминацию train/test: {rep}")
```

Эффект: все четыре задания одной сущности Wikidata остаются в одном сплите;
все четыре показателя контаминации train/test равны нулю.

## 4. Near-duplicate очистка отсутствовала

```diff
--- a/src/clean.py
+++ b/src/clean.py
@@
-from src.dedup import exact_duplicates
+from src.dedup import exact_duplicates, near_duplicates
@@
-    # 5. TODO: сюда просится ещё один шаг дедупликации.
     near: set[int] = set()
+    near_cfg = cfg["near_dup"]
+    if near_cfg["enabled"]:
+        near = set(
+            near_duplicates(
+                [normalize_text(ex.user) for ex in kept],
+                shingle_words=near_cfg["shingle_words"],
+                num_perm=near_cfg["num_perm"],
+                threshold=near_cfg["threshold"],
+            )
+        )
+        kept = [ex for i, ex in enumerate(kept) if i not in near]
```

Эффект: тест очистки удаляет один точный и один почти-дубль. На полном v2
near-duplicate гейт дополнительно удалил одну запись.

## 5. Diversity был предупреждением, а не гейтом

```diff
--- a/src/diversity.py
+++ b/src/diversity.py
@@
+import sys
@@
     if failed:
-        print("diversity: предупреждение — " + "; ".join(failed))
+        print("diversity: гейт закрыт", file=sys.stderr)
+        for message in failed:
+            print(f"  - {message}", file=sys.stderr)
+        raise DiversityError(f"нарушено порогов: {len(failed)}")
```

Эффект: вырожденный fixture завершается ненулевым кодом и печатает четыре
конкретных нарушения; нормальная v2 проходит все пороги.
