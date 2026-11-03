"""End-to-end regression tests: real processes, files and HTTP, no model weights."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def call(*args):
    return subprocess.run([sys.executable, str(ROOT / 'lab.py'), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=45)

def request(port, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'lab'
        result = call('--workspace', self.root, 'init')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.procs = []; self.addCleanup(self.stop)
    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
    def run_lab(self, *args): return call('--workspace', self.root, 'run', *args)
    def server(self, warmup=0):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        proc = subprocess.Popen([sys.executable, str(ROOT/'service.py'), '--workspace', str(self.root),
                                 '--port', str(port), '--warmup-seconds', str(warmup)],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(proc)
        for _ in range(200):
            if proc.poll() is not None: self.fail('service exited unexpectedly')
            try: request(port, '/health'); return port, proc
            except (OSError, urllib.error.URLError): time.sleep(.05)
        self.fail('service failed to start')
    def ready(self, port):
        for _ in range(200):
            if request(port, '/ready')[0] == 200: return
            time.sleep(.05)
        self.fail('service never became ready')
    def test_cache_recovery_and_input_change(self):
        first = self.run_lab('--fail-after-prepare'); self.assertNotEqual(first.returncode, 0)
        second = self.run_lab(); self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn('CACHE_HIT', second.stdout)
        (self.root/'input.txt').write_text('new bytes\n')
        third = self.run_lab(); self.assertEqual(third.returncode, 0, third.stderr)
        self.assertIn('PREPARED', third.stdout)
    def test_incomplete_foreign_and_rejection_preserve_pointer(self):
        before = (self.root/'accepted.json').read_bytes()
        for report in ('report-incomplete.json', 'report-A.json', 'report-reject.json'):
            r = self.run_lab('--report', report)
            self.assertNotEqual(r.returncode, 0, report)
            self.assertEqual((self.root/'accepted.json').read_bytes(), before)
    def test_duplicate_publication(self):
        for _ in range(2):
            r = self.run_lab(); self.assertEqual(r.returncode, 0, r.stderr)
        events = (self.root/'publications.jsonl').read_text().splitlines()
        self.assertEqual(len(events), 1)
    def test_readiness_blocks_generation(self):
        port, _ = self.server(3)
        self.assertEqual(request(port, '/health')[0], 200)
        self.assertEqual(request(port, '/ready')[0], 503)
        self.assertEqual(request(port, '/generate', {'prompt':'hello','max_new_tokens':1})[0], 503)
        self.assertEqual(request(port, '/debug')[1]['calls'], 0)
        self.ready(port)
    def test_invalid_requests_never_call_backend(self):
        port, _ = self.server(); self.ready(port)
        for payload in ({'prompt':'','max_new_tokens':1}, {'prompt':'ok','max_new_tokens':0},
                        {'prompt':'ok','max_new_tokens':100000}, {'prompt':'ok','max_new_tokens':'2'}):
            self.assertTrue(400 <= request(port, '/generate', payload)[0] < 500, payload)
        self.assertEqual(request(port, '/debug')[1]['calls'], 0)
        self.assertEqual(request(port, '/generate', {'prompt':'ok','max_new_tokens':32})[0], 200)
        self.assertEqual(request(port, '/debug')[1]['calls'], 1)
    def test_actual_loaded_version_and_rollback(self):
        port, proc = self.server(); self.ready(port)
        ask = lambda: request(port, '/generate', {'prompt':'ok','max_new_tokens':32})[1]['model_id']
        self.assertEqual(ask(), 'fixture-A')
        r = self.run_lab(); self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(ask(), 'fixture-A')
        proc.terminate(); proc.wait(timeout=5)
        port, proc = self.server(); self.ready(port)
        self.assertEqual(request(port, '/generate', {'prompt':'ok','max_new_tokens':32})[1]['model_id'], 'fixture-B')
        # Roll back to previously accepted A using the same verified publication path.
        rollback=self.run_lab('--manifest','manifest-A.json','--report','report-A.json')
        self.assertEqual(rollback.returncode,0,rollback.stderr)
        self.assertEqual(json.loads((self.root/'publications.jsonl').read_text().splitlines()[-1])['model_id'],'fixture-A')
        proc.terminate(); proc.wait(timeout=5)
        port, _ = self.server(); self.ready(port)
        self.assertEqual(request(port, '/generate', {'prompt':'ok','max_new_tokens':32})[1]['model_id'], 'fixture-A')

    def test_modified_artifact_rejected(self):
        before=(self.root/'accepted.json').read_bytes()
        (self.root/'model-B.txt').write_text('tampered fixture\n')
        r=self.run_lab()
        self.assertNotEqual(r.returncode,0)
        self.assertEqual((self.root/'accepted.json').read_bytes(),before)

    def test_pointer_journal_crash_recovery(self):
        r=self.run_lab('--fail-after-pointer'); self.assertNotEqual(r.returncode,0)
        self.assertEqual(json.loads((self.root/'accepted.json').read_text())['model_id'],'fixture-B')
        self.assertFalse((self.root/'publications.jsonl').exists())
        for _ in range(2):
            r=self.run_lab(); self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len((self.root/'publications.jsonl').read_text().splitlines()),1)

if __name__ == '__main__': unittest.main(verbosity=2)
