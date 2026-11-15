"""Recreate the fixed classification indexes from official COCO annotations.

Only frozen image IDs are distributed. Images and annotations stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base = args.data_root.resolve()
    out = args.output.resolve()
    manifest = read(ROOT / 'data/splits/manifest.json')
    metadata = dict(schema_version=1, data_root=str(base), task='coco_multilabel',
                    categories=manifest['categories'], label_policy=manifest['label_policy'],
                    split_protocol=manifest['split_id'], splits={})
    # Validate all input locations before creating a partial output directory.
    directories = {}
    for name in ('train2017', 'val2017'):
        candidates = [base / 'images' / name, base / name]
        directories[name] = next((x for x in candidates if x.is_dir()), None)
        if directories[name] is None:
            raise FileNotFoundError(f'Missing {name} image directory under {base}')
        if not (base / 'annotations' / f'instances_{name}.json').is_file():
            raise FileNotFoundError(f'Missing instances_{name}.json')
    out.mkdir(parents=True, exist_ok=False)
    cat_index = {c['id']: i for i, c in enumerate(manifest['categories'])}
    for source, roles in [('train2017', ['train', 'test']), ('val2017', ['val'])]:
        annotation = base / 'annotations' / f'instances_{source}.json'
        content = read(annotation)
        assert sorted(content['categories'], key=lambda c: c['id']) == manifest['categories']
        images = {x['id']: x['file_name'] for x in content['images']}
        labels = {i: set() for i in images}
        for instance in content['annotations']:
            # Match the frozen experiment indexes: crowd categories are included.
            labels[instance['image_id']].add(cat_index[instance['category_id']])
        for role in roles:
            split = manifest['splits'][role]
            ids_file = ROOT / split['ids_file']
            assert hashlib.sha256(ids_file.read_bytes()).hexdigest() == split['sha256']
            ids = [int(i) for i in ids_file.read_text().splitlines()]
            assert len(ids) == len(set(ids)) == split['count']
            rows = [[i, images[i], sorted(labels[i])] for i in ids]
            path = out / (split['key'] + '.json')
            write(path, rows)
            metadata['splits'][split['key']] = dict(count=len(rows), image_directory=str(directories[source]),
                annotation_file=str(annotation), index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                role=role, samples=[dict(image_id=i, path=str(directories[source] / images[i])) for i in ids[:8]])
        del content, labels, images
    metadata['fingerprint_sha256'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    write(out / 'dataset.json', metadata)
    print(f'Prepared fixed train/val/test indexes: {out}')


if __name__ == '__main__':
    main()
