"""Full COCO image-level labels, stateless augmentation and resumable batches.

This module deliberately does not import the model/Lightning in loader workers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_index(data_root, output):
    root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "data_root": str(root), "task": "coco_multilabel",
              "label_policy": "all annotated categories, including crowd; empty-label images retained",
              "splits": {}}
    ids = []
    for split in ("train2017", "val2017"):
        annotation = root / "annotations" / f"instances_{split}.json"
        data = json.loads(annotation.read_text(encoding="utf-8"))
        categories = sorted(data["categories"], key=lambda c: c["id"])
        if "categories" in result:
            assert categories == result["categories"]
        result["categories"] = categories
        lookup = {c["id"]: i for i, c in enumerate(categories)}
        images = {row["id"]: row for row in data["images"]}
        labels = {key: set() for key in images}
        for row in data["annotations"]:
            labels[row["image_id"]].add(lookup[row["category_id"]])
        records = [[key, images[key]["file_name"], sorted(labels[key])] for key in sorted(images)]
        missing = [name for _, name, _ in records if not (root / "images" / split / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} missing images; first: {missing[:5]}")
        ids.append(set(images))
        index_path = output / f"{split}.json"
        # Compact JSON keeps Windows worker startup small.
        index_path.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
        result["splits"][split] = {"count": len(records), "annotation_sha256": file_hash(annotation),
            "index_sha256": file_hash(index_path), "empty_label_images": sum(not r[2] for r in records),
            "category_positive_images": [sum(i in r[2] for r in records) for i in range(80)],
            "samples": [{"image_id": r[0], "path": str(root / "images" / split / r[1])} for r in records[:8]]}
        print(f"Indexed {split}: {len(records)} images", flush=True)
        del data, images, labels, records
    assert not ids[0].intersection(ids[1]), "Train/validation overlap"
    result["fingerprint_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    write_json(output / "dataset.json", result)
    return result


def decode_audit(index_dir, workers=12):
    """Decode EVERY image; also hash its bytes. Never skip corrupt samples."""
    index_dir = Path(index_dir)
    meta = json.loads((index_dir / "dataset.json").read_text(encoding="utf-8"))
    root = Path(meta["data_root"])
    result = {"dataset_fingerprint": meta["fingerprint_sha256"], "status": "running", "splits": {}}
    start = time.perf_counter()
    def check(item):
        split, row = item
        path = root / "images" / split / row[1]
        try:
            with Image.open(path) as img:
                img.convert("RGB").load()
            return row[0], file_hash(path), None
        except Exception as exc:
            return row[0], None, f"{path}: {exc!r}"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split in ("train2017", "val2017"):
            rows = json.loads((index_dir / f"{split}.json").read_text(encoding="utf-8"))
            digest, failures = hashlib.sha256(), []
            for offset in range(0, len(rows), 2000):
                for image_id, content_hash, error in pool.map(check, [(split, r) for r in rows[offset:offset+2000]]):
                    if error:
                        failures.append(error)
                    else:
                        digest.update(f"{image_id}:{content_hash}\n".encode())
                if offset % 10000 == 0:
                    print(f"Decoded {split}: {min(offset+2000, len(rows))}/{len(rows)}", flush=True)
            result["splits"][split] = {"count": len(rows), "failures": failures,
                                       "ordered_image_content_sha256": digest.hexdigest()}
            write_json(index_dir / "decode_audit.json", result)
    result.update(status="passed" if all(not r["failures"] for r in result["splits"].values()) else "failed",
                  wall_s=time.perf_counter()-start)
    write_json(index_dir / "decode_audit.json", result)
    if result["status"] != "passed":
        raise RuntimeError("Image decode audit failed; see decode_audit.json")
    print(f"Full decode audit passed in {result['wall_s']:.1f}s", flush=True)


class CocoImages(Dataset):
    def __init__(self, index_dir, split, resolution=224, data_seed=42042, augment=False, limit=None):
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "dataset.json").read_text(encoding="utf-8"))
        default_directory = Path(meta["data_root"]) / "images" / split
        # Frozen holdouts can share the original image folder while using distinct indexes.
        self.directory = Path(meta.get("splits", {}).get(split, {}).get("image_directory", default_directory))
        self.rows = json.loads((index_dir / f"{split}.json").read_text(encoding="utf-8"))
        if limit:
            self.rows = self.rows[:limit]
        self.resolution, self.seed, self.augment = resolution, data_seed, augment
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        image_id, filename, positive = self.rows[index]
        with Image.open(self.directory / filename) as image:
            image = image.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
            if self.augment:
                rng = random.Random(self.seed + 1000003 * epoch + 9176 * image_id)
                if rng.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                operations = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
                rng.shuffle(operations)
                for operation in operations:
                    image = operation(image).enhance(rng.uniform(0.8, 1.2))
            array = np.asarray(image, dtype=np.float32).copy()
        x = (torch.from_numpy(array).permute(2, 0, 1) / 255. - self.mean) / self.std
        y = torch.zeros(80)
        y[positive] = 1.
        return x, y, image_id


class EpochBatches(Sampler):
    def __init__(self, count, batch_size, seed, epoch=0, start_batch=0):
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.epoch, self.start_batch = epoch, start_batch

    def indices(self):
        return torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + 50 + self.epoch))

    def __iter__(self):
        order = self.indices().tolist()
        for offset in range(self.start_batch*self.batch_size, self.count, self.batch_size):
            yield [(self.epoch, i) for i in order[offset:offset+self.batch_size]]

    def __len__(self):
        return math.ceil(self.count/self.batch_size)-self.start_batch


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(dataset, config, sampler=None):
    workers = config["loader_workers"]
    options = dict(num_workers=workers, pin_memory=True, worker_init_fn=worker_init,
                   generator=torch.Generator().manual_seed(config["data_seed"]+77))
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    if sampler is None:
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, **options)
    return DataLoader(dataset, batch_sampler=sampler, **options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/datasets/coco2017")
    parser.add_argument("--output", default="outputs/coco-full-index")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    prepare_index(args.data_root, args.output)
    if args.audit:
        decode_audit(args.output)
