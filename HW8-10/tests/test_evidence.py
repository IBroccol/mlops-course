"""The baseline decision must remain tied to the original full reports and raw output."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import release

ROOT=Path(__file__).resolve().parents[1]

class EvidenceTests(unittest.TestCase):
    def test_saved_evidence_and_tampered_raw(self):
        report=json.loads((ROOT/'evidence/release-report.json').read_text())
        manifest=json.loads((ROOT/'evidence/manifest-baseline.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            for role in ('baseline','candidate'):
                for kind in ('report','outputs'):
                    source=ROOT.parent/'HW6-7'/'docs'/'reports'/f"{role}-{'eval' if kind=='report' else 'outputs'}.json"
                    target=Path(temp)/source.name; shutil.copyfile(source,target)
                    report['source_reports'][role][kind]=str(target)
            release.verify_evidence(report,manifest)
            target=Path(report['source_reports']['baseline']['outputs']); target.write_text('[]')
            with self.assertRaises(ValueError): release.verify_evidence(report,manifest)
