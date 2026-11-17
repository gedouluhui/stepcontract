import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_demos_use_real_training_evidence(self):
        expected = {'healthy': ('PASS', 0), 'accumulation': ('PASS', 0),
                    'missing-optimizer': ('FAIL', 1), 'early-zero': ('FAIL', 1), 'nonfinite': ('FAIL', 1)}
        with tempfile.TemporaryDirectory() as tmp:
            for name, (status, code) in expected.items():
                with self.subTest(case=name):
                    out = Path(tmp) / (name + '.json')
                    p = subprocess.run([sys.executable, '-m', 'stepcontract', 'demo', '--case', name,
                                        '--json', str(out), '--html', str(out.with_suffix('.html'))],
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, code, p.stderr)
                    self.assertTrue(out.is_file(), 'CLI must produce a real report')
                    r = json.loads(out.read_text())
                    self.assertEqual(r['status'], status)
                    self.assertGreater(len(r['events']), 2)
                    self.assertEqual(r['summary']['completed_steps'], 1)
                    if status == 'FAIL':
                        self.assertTrue(r['findings'])

    def test_safe_html(self):
        self.assertIsNotNone(importlib.util.find_spec('stepcontract.report'), 'HTML renderer must exist')
        from stepcontract.report import to_html
        value = '<img src=x onerror=alert(1)>'
        text = to_html({'status': 'FAIL', 'scope': value, 'events': [], 'parameters': [],
                        'findings': [{'rule': value, 'parameter': value, 'detail': value}], 'summary': {}})
        self.assertNotIn('<img', text)
        self.assertIn('&lt;img', text)
        self.assertNotIn('<script', text)

    def test_output_alias_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'one.txt'
            p = subprocess.run([sys.executable, '-m', 'stepcontract', 'demo', '--case', 'healthy',
                                '--json', str(out), '--html', str(out)], capture_output=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
