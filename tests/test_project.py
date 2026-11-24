import ast, glob, importlib, json, os, re, subprocess, sys, unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SIM = os.path.join(ROOT, 'risuav', 'sim')


def sources():
    for pat in ('risuav/**/*.py', 'tests/*.py'):
        yield from glob.glob(os.path.join(ROOT, pat), recursive=True)


class ProjectTests(unittest.TestCase):
    def test_configs_are_valid_json(self):
        for f in glob.glob(os.path.join(ROOT, 'configs', '*.json')) + glob.glob(os.path.join(ROOT, '.vscode', '*.json')):
            with open(f, encoding='utf-8') as fh:
                json.load(fh)

    def test_no_machine_specific_paths(self):
        bad = re.compile(r"(/home/\w+|/mnt/|/tmp/|[A-Za-z]:\\\\Users|/Users/\w+)")
        hits = []
        for f in sources():
            with open(f, encoding='utf-8') as fh:
                for i, line in enumerate(fh, 1):
                    if bad.search(line) and 'bad = re.compile' not in line:
                        hits.append(f'{os.path.relpath(f, ROOT)}:{i}: {line.strip()[:80]}')
        self.assertEqual(hits, [], 'machine-specific paths found')

    def test_sources_parse(self):
        for f in sources():
            with open(f, encoding='utf-8') as fh:
                ast.parse(fh.read(), f)

    def test_library_modules_import(self):
        sys.path.insert(0, SIM)
        for m in ('physics', 'env', 'batch', 'model', 'fastmodel', 'model_mr', 'model_rx', 'plan', 'plan_mr', 'build_exact',
                  'coupling', 'env_mr', 'rx_common', 'rx_paths', 'param_override'):
            importlib.import_module(m)

    def test_paths_are_inside_the_project_by_default(self):
        env = {k: v for k, v in os.environ.items() if k not in ('RISUAV_WORK', 'MR_ROOT', 'PRELIM_ROOT', 'RUNS_DIR', 'RESULTS_DIR')}
        out = subprocess.run([sys.executable, '-c', 'import rx_paths as p; print(p.MR_ROOT); print(p.RUNS_DIR); print(p.RESULTS_DIR)'],
                             cwd=SIM, env=env, capture_output=True, text=True, check=True).stdout.split()
        for p in out:
            self.assertTrue(os.path.abspath(p).startswith(ROOT), p)

    def test_cli_parser_and_dry_run(self):
        for args in (['-h'], ['figures', '--dry-run'], ['reproduce', '--dry-run', '--skip-training', '--skip-sensitivity']):
            r = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'risuav'] + args, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_published_artifacts(self):
        for p in ('facade', 'ground'):
            for s in (0, 1, 2):
                self.assertTrue(os.path.exists(os.path.join(ROOT, 'artifacts', 'checkpoints', f'proposed_{p}_s{s}', 'best.pt')))
                f = os.path.join(ROOT, 'artifacts', 'results', 'main', f'{p}_s{s}_best.jsonl')
                with open(f, encoding='utf-8') as fh:
                    rows = [json.loads(line) for line in fh]
                self.assertEqual(len(rows), 120)
                for r in rows:
                    if r.get('servable_ris'):
                        self.assertTrue(all(r.get(m) for m in ('proposed', 'random', 'coverage')))
        for tag in ('N16', 'N24', 'N40', 'D10', 'D20', 'D80'):
            for p in ('facade', 'ground'):
                with open(os.path.join(ROOT, 'artifacts', 'results', f'sens_{tag}', f'{p}_s0_best.jsonl'), encoding='utf-8') as fh:
                    rows = [json.loads(line) for line in fh]
                self.assertEqual(len(rows), 40)
                self.assertTrue(all(all(r.get(m) for m in ('proposed', 'none', 'random', 'coverage')) for r in rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
