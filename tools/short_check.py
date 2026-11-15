"""Run a short train/resume/evaluate/benchmark cycle on local COCO images."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--group', default='fm_pcd_c2')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/short_check')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)
    index = root/'index'
    run('prepare_data.py', '--data-root', args.data_root.resolve(), '--output', index)
    run('experiments.py', 'prepare', '--group', args.group, '--index', index,
        '--run', root/'run', '--smoke')
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    config_path = root/'run/config.json'
    config = json.loads(config_path.read_text(encoding='utf8'))
    config['loader_workers'] = args.workers
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    common = ['--group', args.group, '--run', root/'run']
    run('experiments.py', 'train', *common, '--max-steps', '1')
    run('experiments.py', 'train', *common, '--resume')
    run('experiments.py', 'evaluate', *common, '--split', 'val')
    run('experiments.py', 'benchmark', *common, '--rounds', '1', '--iterations', '10', '--trace')
    print('Short workflow completed:', root)

if __name__ == '__main__':
    main()
