"""Стадия collect: снимок Wikidata -> русскоязычный chat JSONL."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.config import load_params, source_files


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать системный промпт, не используя salted hash()."""
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def _read_sources(paths: list[Path]) -> tuple[list[dict], int]:
    """Прочитать снимки, проверить контракт и удалить повторные entity_id."""
    rows: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    common = {"entity_id", "kind", "name", "population", "source_url"}
    by_kind = {
        "city": {"country", "admin_region", "latitude", "longitude"},
        "country": {"capital", "continent", "area_km2"},
    }
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: невалидный JSON: {exc.msg}") from exc
                kind = row.get("kind")
                required = common | by_kind.get(kind, set())
                missing = required - set(row)
                if kind not in by_kind or missing:
                    raise SystemExit(
                        f"{path}:{lineno}: kind={kind!r}, отсутствуют поля {sorted(missing)}"
                    )
                qid = str(row["entity_id"])
                if qid in seen:
                    duplicates += 1
                    continue
                seen.add(qid)
                rows.append(row)
    return rows, duplicates


def _population(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _coordinate(value: float, positive: str, negative: str) -> str:
    suffix = positive if value >= 0 else negative
    return f"{abs(value):.5f}° {suffix}"


def _city_card(row: dict) -> str:
    admin = row["admin_region"] or "не указан"
    return (
        f"Карточка источника Wikidata ({row['entity_id']}):\n"
        f"объект — {row['name']}; тип — город; страна — {row['country']}; "
        f"административный регион — {admin}; население — {_population(row['population'])}; "
        f"широта — {row['latitude']}; долгота — {row['longitude']}."
    )


def _country_card(row: dict) -> str:
    area = f"{row['area_km2']:,.2f}".replace(",", " ")
    return (
        f"Карточка источника Wikidata ({row['entity_id']}):\n"
        f"объект — {row['name']}; тип — страна; столица — {row['capital']}; "
        f"континент — {row['continent']}; население — {_population(row['population'])}; "
        f"площадь — {area} км²."
    )


def _city_example(row: dict, task: str, snapshot_date: str) -> tuple[str, str]:
    admin = row["admin_region"] or "в снимке не указан"
    latitude = _coordinate(row["latitude"], "с. ш.", "ю. ш.")
    longitude = _coordinate(row["longitude"], "в. д.", "з. д.")
    questions = {
        "location": "В какой стране и каком административном регионе находится объект?",
        "population": "Какое население указано для объекта? Не представляй число как текущее.",
        "coordinates": "Запиши координаты объекта с обозначением полушарий.",
        "summary": "Составь краткую географическую справку только по данным карточки.",
    }
    answers = {
        "location": (
            f"{row['name']} находится в стране {row['country']}; административный регион — "
            f"{admin}. Источник: Wikidata, {row['entity_id']}."
        ),
        "population": (
            f"В зафиксированном снимке Wikidata от {snapshot_date} для города {row['name']} "
            f"указано население {_population(row['population'])} человек. Это значение снимка, "
            "а не утверждение об актуальной численности на сегодняшний день."
        ),
        "coordinates": (
            f"Координаты города {row['name']}: {latitude}, {longitude}. Они получены из "
            f"структурированного свойства координат объекта {row['entity_id']} в Wikidata."
        ),
        "summary": (
            f"{row['name']} — город в стране {row['country']}. Административный регион: "
            f"{admin}. В снимке Wikidata от {snapshot_date} указано население "
            f"{_population(row['population'])} человек. Координаты: {latitude}, {longitude}. "
            f"Идентификатор источника — {row['entity_id']}; данные следует воспринимать как "
            "состояние источника на дату снимка."
        ),
    }
    return f"{_city_card(row)}\n\nЗадание: {questions[task]}", answers[task]


def _country_example(row: dict, task: str, snapshot_date: str) -> tuple[str, str]:
    area = f"{row['area_km2']:,.2f}".replace(",", " ")
    questions = {
        "location": "На каком континенте находится страна и как называется её столица?",
        "population": "Какие население и площадь указаны в карточке? Добавь оговорку о снимке.",
        "coordinates": "Выдели столицу страны и не добавляй сведений, которых нет в карточке.",
        "summary": "Составь краткую справку о стране только по данным карточки.",
    }
    answers = {
        "location": (
            f"{row['name']} относится к континенту «{row['continent']}»; столица страны — "
            f"{row['capital']}. Источник: Wikidata, {row['entity_id']}."
        ),
        "population": (
            f"В снимке Wikidata от {snapshot_date} для страны {row['name']} указано население "
            f"{_population(row['population'])} человек и площадь {area} км². Эти показатели "
            "относятся к зафиксированной версии источника и могут измениться после её обновления."
        ),
        "coordinates": (
            f"Столица страны {row['name']} — {row['capital']}. Карточка не содержит сведений "
            f"о координатах столицы, поэтому добавлять их нельзя. Идентификатор: {row['entity_id']}."
        ),
        "summary": (
            f"{row['name']} — страна на континенте «{row['continent']}» со столицей "
            f"{row['capital']}. В снимке Wikidata от {snapshot_date} указаны население "
            f"{_population(row['population'])} человек и площадь {area} км². Источник фактов — "
            f"структурированная карточка {row['entity_id']}; сведения не следует автоматически "
            "считать актуальными после даты получения снимка."
        ),
    }
    return f"{_country_card(row)}\n\nЗадание: {questions[task]}", answers[task]


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    prompts = cfg["system_prompts"]
    tasks = cfg["tasks"]
    if len(prompts) < 3:
        raise SystemExit("collect.system_prompts: нужно не меньше трёх вариантов")

    sources = source_files(params)
    rows, source_duplicates = _read_sources(sources)
    snapshot = json.loads(Path(cfg["snapshot_metadata"]).read_text(encoding="utf-8"))
    snapshot_date = snapshot["retrieved_at_utc"][:10]
    started = time.perf_counter()
    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)
    kinds: dict[str, int] = {"city": 0, "country": 0}
    prompts_used: set[str] = set()

    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            kinds[row["kind"]] += 1
            for task in tasks:
                example_id = f"geo-{row['entity_id']}-{task}"
                system = pick_prompt(example_id, prompts)
                prompts_used.add(system)
                if row["kind"] == "city":
                    user, assistant = _city_example(row, task, snapshot_date)
                else:
                    user, assistant = _country_example(row, task, snapshot_date)
                record = {
                    "id": example_id,
                    "topic": f"{row['kind']}/{row['entity_id']}",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = round(time.perf_counter() - started, 2)
    metrics = {
        "version": cfg["version"],
        "snapshot_date": snapshot_date,
        "source_files": len(sources),
        "source_entities": len(rows),
        "source_duplicates_skipped": source_duplicates,
        "cities": kinds["city"],
        "countries": kinds["country"],
        "tasks_per_entity": len(tasks),
        "rows_written": len(rows) * len(tasks),
        "system_prompt_variants": len(prompts_used),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"collect: {cfg['version']}, сущностей {len(rows)} "
        f"(города {kinds['city']}, страны {kinds['country']}), "
        f"примеров {metrics['rows_written']}, {elapsed} с -> {out}"
    )


if __name__ == "__main__":
    main()
