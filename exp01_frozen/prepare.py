"""Prepare a group-disjoint VisDrone diagnostic without modifying source data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import zipfile

from PIL import Image
import yaml

from common import NAMES, ROLES, arguments, config, digest, write_json


def prepare(cfg, out):
    if (out / "manifest.jsonl").exists():
        raise ValueError("Run already prepared; use a new --run directory to change the protocol.")
    if len(cfg["observed_classes"]) != 3:
        raise ValueError("This pilot expects exactly three observed classes.")
    root = Path(cfg["visdrone_root"])
    records = []
    for split, filename in [("train", "VisDrone2019-DET-train.zip"),
                            ("val", "VisDrone2019-DET-val.zip"),
                            ("test", "VisDrone2019-DET-test-dev.zip")]:
        with zipfile.ZipFile(root.parent / filename) as archive:
            paths = sorted(n for n in archive.namelist() if "annotations/" in n and n.endswith(".txt"))
            for name in paths:
                stem = Path(name).stem
                gt, ignored = [], []
                for line in archive.read(name).decode("utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    x, y, w, h, score, cls, truncation, occlusion = map(float, line.split(",")[:8])
                    if w <= 0 or h <= 0:
                        continue
                    box = [x, y, x + w, y + h]
                    if int(score) == 0 or int(cls) not in range(1, 11):
                        ignored.append(box)
                    else:
                        gt.append({"box": box, "cls": int(cls) - 1,
                                   "occlusion": int(occlusion), "truncation": int(truncation)})
                records.append({"id": f"{split}/{stem}", "source_split": split,
                                "group": stem.split("_")[0], "image": str(root / "images" / split / f"{stem}.jpg"),
                                "gt": gt, "ignore": ignored})
    if not records:
        raise ValueError("No raw annotation files found.")
    groups = {s: {r["group"] for r in records if r["source_split"] == s} for s in ("train", "val", "test")}
    if groups["val"] & groups["test"]:
        raise ValueError("Official validation/test groups overlap; a new split is required.")
    excluded = groups["train"] & (groups["val"] | groups["test"])
    eligible = sorted(groups["train"] - excluded)
    random.Random(cfg["seed"]).shuffle(eligible)
    n_train = int(len(eligible) * cfg["detector_train_fraction"])
    n_val = int(len(eligible) * cfg["detector_val_fraction"])
    if not (n_train > 0 and n_val > 0 and n_train + n_val < len(eligible)):
        raise ValueError("Group fractions produce an empty split.")
    assignments = {g: "det_train" if i < n_train else "det_val" if i < n_train + n_val else "quality_fit"
                   for i, g in enumerate(eligible)}
    out.mkdir(parents=True, exist_ok=True)
    data = out / "detector_data"
    for i, r in enumerate(records):
        split, group = r["source_split"], r["group"]
        r["role"] = assignments.get(group, "excluded") if split == "train" else "quality_val" if split == "val" else "test"
        with Image.open(r["image"]) as image:
            r["width"], r["height"] = image.size
        if r["role"] in ("det_train", "det_val"):
            part = "train" if r["role"] == "det_train" else "val"
            image_link = data / "images" / part / Path(r["image"]).name
            label = data / "labels" / part / (Path(r["image"]).stem + ".txt")
            image_link.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            if not image_link.is_symlink():
                image_link.symlink_to(r["image"])
            lines = []
            for g in r["gt"]:
                x1, y1, x2, y2 = g["box"]
                x1, x2 = max(0, x1), min(r["width"], x2)
                y1, y2 = max(0, y1), min(r["height"], y2)
                if x2 > x1 and y2 > y1:
                    values = [(x1 + x2) / (2 * r["width"]), (y1 + y2) / (2 * r["height"]),
                              (x2 - x1) / r["width"], (y2 - y1) / r["height"]]
                    lines.append(str(g["cls"]) + " " + " ".join(f"{v:.8f}" for v in values))
            label.write_text("\n".join(lines) + ("\n" if lines else ""))
        if (i + 1) % 1000 == 0:
            print(f"Prepared {i + 1}/{len(records)} images", flush=True)
    with (out / "manifest.jsonl").open("w") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    for role in ROLES:
        selected = [r for r in records if r["role"] == role]
        (out / f"{role}.txt").write_text("".join(r["image"] + "\n" for r in selected))
    (out / "detector.yaml").write_text(yaml.safe_dump({"path": str(data.resolve()), "train": "images/train",
                                                     "val": "images/val", "names": dict(enumerate(NAMES))}, sort_keys=False))
    architecture = yaml.safe_load((Path(cfg["detector_repo"]) / "ultralytics/cfg/models/11/yolo11.yaml").read_text())
    architecture.update({"nc": 10, "scale": "n"})
    (out / "detector_arch.yaml").write_text(yaml.safe_dump(architecture, sort_keys=False))
    write_json(out / "config.json", cfg)
    summary = {"images": dict(Counter(r["role"] for r in records)),
               "groups": {role: len({r["group"] for r in records if r["role"] == role}) for role in ROLES},
               "excluded_train_groups": sorted(excluded), "manifest_sha256": digest(out / "manifest.jsonl"),
               "group_rule": "filename prefix before first underscore; conservative proxy, not a verified scene ID",
               "annotation_rule": "score=0 or class outside 1..10 becomes ignore rectangle",
               "uavdt_role": "reserved for stage-2 replication; not pooled in this pilot"}
    write_json(out / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    prepare(config(args.config), args.run.resolve())
