#!/usr/bin/env python3
"""Compare every original cell and figure with a pinned upstream Git commit."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import nbformat

ROOT = Path(__file__).resolve().parents[1]
BASE = 'd953252941ac80e3e2856e1e31f4aff38ba43f1b'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    stats = {'baseline_commit': BASE, 'notebooks': 0, 'original_cells': 0,
             'original_code_cells': 0, 'stored_outputs': 0, 'unchanged_figures': 0}
    for name in git('ls-tree', '-r', '--name-only', BASE).decode().splitlines():
        path = ROOT / name
        if name.startswith('figures/'):
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(git('show', f'{BASE}:{name}')).digest(), name
            stats['unchanged_figures'] += 1
        if not name.endswith('.ipynb'):
            continue
        old = json.loads(git('show', f'{BASE}:{name}'))
        new = json.loads(path.read_text())
        nbformat.validate(nbformat.from_dict(copy.deepcopy(new)))
        cells = [copy.deepcopy(c) for c in new['cells'] if not any(
            t.startswith('dataerai-managed-') for t in c.get('metadata', {}).get('tags', []))]
        assert len(cells) == len(old['cells']), name
        for before, after in zip(old['cells'], cells):
            source = ''.join(after['source'])
            if before['cell_type'] == 'code' and ''.join(before['source']).strip():
                assert source.startswith('%%dataerai\n'), name
                source = source[len('%%dataerai\n'):]
                stats['original_code_cells'] += 1
            assert source == ''.join(before['source']), name
            after['source'] = before['source']
            if 'id' not in before:
                after.pop('id', None)
            assert after == before, name  # includes outputs, execution counts, metadata, attachments
            stats['original_cells'] += 1
            stats['stored_outputs'] += len(before.get('outputs', []))
        stats['notebooks'] += 1
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == '__main__':
    main()
