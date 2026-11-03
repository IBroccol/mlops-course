#!/bin/sh
# All experiments use an isolated synthetic workspace; no model or production release is changed.
set -u
export PREFECT_HOME="$PWD/.prefect" PREFECT_SERVER_ANALYTICS_ENABLED=false
ROOT="$PWD/.lab-flow-v3"
if [ -e "$ROOT" ]; then echo "Remove the old demo workspace or choose a new one" >&2; exit 1; fi
.venv/bin/python lab.py --workspace "$ROOT" init
run() { .venv/bin/python flow.py --workspace "$ROOT" "$@"; }
echo '--- fail after preparation ---'
run --fail-after-prepare; echo "failure_exit=$?"
echo '--- resume in a new process ---'
run; echo "resume_exit=$?"
echo '--- same path, new content ---'
printf 'changed input\n' > "$ROOT/input.txt"
run; echo "changed_exit=$?"
echo '--- rejected report and pointer checksum ---'
shasum -a 256 "$ROOT/accepted.json"
run --report report-reject.json; echo "reject_exit=$?"
shasum -a 256 "$ROOT/accepted.json"
echo '--- pointer/journal crash and repeat ---'
# A fresh release identity guarantees an absent journal entry at the crash point.
.venv/bin/python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'report-B.json'; r=json.loads(p.read_text()); r['evaluation_run_id']='fixture-eval-B-second'; p.write_text(json.dumps(r))
PY
run --fail-after-pointer; echo "pointer_crash_exit=$?"
run; echo "recovery_exit=$?"
run; echo "repeat_exit=$?"
cat "$ROOT/publications.jsonl"
