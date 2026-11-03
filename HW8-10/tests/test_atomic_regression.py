"""A simulated replace failure must leave the accepted bytes untouched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import release

class AtomicWriteRegression(unittest.TestCase):
    def test_replace_failure_keeps_old_bytes(self):
        manifest=json.loads((Path(__file__).resolve().parents[1]/'examples/manifest.example.json').read_text())
        report=json.loads((Path(__file__).resolve().parents[1]/'examples/report.example.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'accepted.json'; original=b'{"old": true}\n'; path.write_bytes(original)
            with patch('os.replace',side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError): release.publish(manifest,report,path)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(td).glob('*.tmp')),[])
