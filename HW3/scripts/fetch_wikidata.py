#!/usr/bin/env python3
"""Скачать и зафиксировать исходный снимок географических фактов Wikidata.

Скрипт не входит в обычный ``dvc repro``: внешний источник со временем
меняется и может быть временно недоступен. После скачивания ``sources/data``
версионируется DVC, а все последующие стадии работают только со снимком.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sources" / "data"
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "mlops-course-hw3/1.0 (educational dataset; github.com/IBroccol/mlops-course)"

CITY_QUERY = """
SELECT ?item ?itemLabel ?country ?countryLabel ?admin ?adminLabel ?population ?coord WHERE {
  ?item wdt:P31 wd:Q515;
        wdt:P17 ?country;
        wdt:P1082 ?population;
        wdt:P625 ?coord;
        rdfs:label ?itemLabel.
  ?country rdfs:label ?countryLabel.
  FILTER(LANG(?itemLabel) = "ru")
  FILTER(LANG(?countryLabel) = "ru")
  OPTIONAL {
    ?item wdt:P131 ?admin.
    ?admin rdfs:label ?adminLabel.
    FILTER(LANG(?adminLabel) = "ru")
  }
}
ORDER BY DESC(?population)
LIMIT 1400
"""

COUNTRY_QUERY = """
SELECT ?item ?itemLabel ?capital ?capitalLabel ?continent ?continentLabel ?population ?area WHERE {
  ?item wdt:P31 wd:Q3624078;
        wdt:P36 ?capital;
        wdt:P30 ?continent;
        wdt:P1082 ?population;
        wdt:P2046 ?area;
        rdfs:label ?itemLabel.
  ?capital rdfs:label ?capitalLabel.
  ?continent rdfs:label ?continentLabel.
  FILTER(LANG(?itemLabel) = "ru")
  FILTER(LANG(?capitalLabel) = "ru")
  FILTER(LANG(?continentLabel) = "ru")
}
ORDER BY DESC(?population)
LIMIT 600
"""

POINT = re.compile(r"^Point\(([-+0-9.eE]+) ([-+0-9.eE]+)\)$")


def _query(sparql: str, attempts: int = 4) -> list[dict]:
    """Выполнить SPARQL POST с ограниченным повтором временных ошибок."""
    body = urlencode({"query": sparql, "format": "json"}).encode("utf-8")
    request = Request(
        ENDPOINT,
        data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)["results"]["bindings"]
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise SystemExit(f"Wikidata query failed after {attempts} attempts: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _value(binding: dict, key: str, default: str = "") -> str:
    return binding.get(key, {}).get("value", default).strip()


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _integer(value: str) -> int:
    return int(float(value))


def _cities(bindings: list[dict]) -> list[dict]:
    """Дедуплицировать города, оставив запись с максимальным населением."""
    rows: dict[str, dict] = {}
    for item in bindings:
        match = POINT.match(_value(item, "coord"))
        if not match:
            continue
        qid = _qid(_value(item, "item"))
        candidate = {
            "entity_id": qid,
            "kind": "city",
            "name": _value(item, "itemLabel"),
            "country": _value(item, "countryLabel"),
            "admin_region": _value(item, "adminLabel"),
            "population": _integer(_value(item, "population")),
            "longitude": round(float(match.group(1)), 5),
            "latitude": round(float(match.group(2)), 5),
            "source_url": f"https://www.wikidata.org/wiki/{qid}",
        }
        previous = rows.get(qid)
        if previous is None or candidate["population"] > previous["population"]:
            rows[qid] = candidate
    return sorted(rows.values(), key=lambda row: (-row["population"], row["entity_id"]))


def _countries(bindings: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for item in bindings:
        qid = _qid(_value(item, "item"))
        candidate = {
            "entity_id": qid,
            "kind": "country",
            "name": _value(item, "itemLabel"),
            "capital": _value(item, "capitalLabel"),
            "continent": _value(item, "continentLabel"),
            "population": _integer(_value(item, "population")),
            "area_km2": round(float(_value(item, "area")), 2),
            "source_url": f"https://www.wikidata.org/wiki/{qid}",
        }
        previous = rows.get(qid)
        if previous is None or candidate["population"] > previous["population"]:
            rows[qid] = candidate
    return sorted(rows.values(), key=lambda row: (-row["population"], row["entity_id"]))


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cities = _cities(_query(CITY_QUERY))
    countries = _countries(_query(COUNTRY_QUERY))
    if len(cities) < 650:
        raise SystemExit(f"получено только {len(cities)} уникальных городов, нужно минимум 650")
    if len(countries) < 150:
        raise SystemExit(f"получено только {len(countries)} стран, нужно минимум 150")

    v1 = cities[:400]
    v2 = cities[400:700] + countries
    _write(OUT_DIR / "world_v1.jsonl", v1)
    _write(OUT_DIR / "world_v2.jsonl", v2)
    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "endpoint": ENDPOINT,
        "license": "CC0-1.0",
        "v1_entities": len(v1),
        "v2_additional_entities": len(v2),
        "cities_available": len(cities),
        "countries_available": len(countries),
    }
    (OUT_DIR / "snapshot.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wikidata snapshot: v1 {len(v1)} городов; "
        f"расширение v2 {len(v2)} сущностей ({len(countries)} стран)"
    )


if __name__ == "__main__":
    main()
