"""Check all 14 model interfaces with precompiled kernels."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    groups = json.loads((ROOT/'configs/groups.json').read_text(encoding='utf8'))['groups']
    env = dict(os.environ, TORCHDYNAMO_DISABLE='1', PYTHONUTF8='1')
    for group in groups:
        subprocess.run([sys.executable, str(ROOT/'experiments.py'), 'smoke',
            '--group', group['id'], '--output', str(ROOT/'outputs/verification'/f"{group['id']}.json")],
            cwd=ROOT, env=env, check=True)
    print('14 model checks passed.')

if __name__ == '__main__':
    main()
