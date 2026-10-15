"""Shared geometry and features. Feature functions never receive hidden GT."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]
ROLES = ("det_train", "det_val", "quality_fit", "quality_val", "test")


def arguments(parser):
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--run", type=Path, default=HERE / "runs/v1")
    return parser


def config(path):
    cfg = json.loads(path.read_text())
    if cfg["target_class"] in cfg["observed_classes"]:
        raise ValueError("Target class must be hidden.")
    return cfg


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boxes(values):
    return np.asarray(values, dtype=float).reshape(-1, 4)


def area(b):
    return np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)


def ious(a, b):
    a, b = boxes(a), boxes(b)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=-1)
    return intersection / np.maximum(area(a)[:, None] + area(b)[None, :] - intersection, 1e-12)


def match(predictions, gt, threshold):
    """Fixed score order, same-class greedy one-to-one match; unmatched = -1."""
    result = np.full(len(predictions), -1, dtype=int)
    overlaps = ious([p["box"] for p in predictions], [g["box"] for g in gt])
    used = set()
    order = sorted(range(len(predictions)), key=lambda i: (-predictions[i]["score"], i))
    for i in order:
        eligible = [j for j, g in enumerate(gt)
                    if j not in used and g["cls"] == predictions[i]["cls"]
                    and overlaps[i, j] >= threshold]
        if eligible:
            j = max(eligible, key=lambda j: overlaps[i, j])
            result[i] = j
            used.add(j)
    return result, overlaps


def ignore_filter(items, regions, width, height, threshold):
    """Union of ignore rectangles, raster IoA. Also drops degenerate boxes.

    Uses floor/ceil half-open pixels, not the MATLAB toolkit's rounding.
    This is an explicitly fixed diagnostic protocol, not official AP code.
    """
    if not items:
        return []
    integral = None
    if regions:
        mask = np.zeros((height, width), dtype=np.int32)
        for rect in regions:
            x1, y1 = np.floor(rect[:2]).astype(int)
            x2, y2 = np.ceil(rect[2:]).astype(int)
            mask[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)] = 1
        integral = np.pad(mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
    kept = []
    for item in items:
        b = np.clip(item["box"], [0, 0, 0, 0], [width, height, width, height])
        if area(b) <= 0:
            continue
        x1, y1 = np.floor(b[:2]).astype(int)
        x2, y2 = np.ceil(b[2:]).astype(int)
        if integral is not None:
            covered = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
            if covered / max((x2 - x1) * (y2 - y1), 1) >= threshold:
                continue
        kept.append({**item, "box": b.tolist()})
    return kept


def relationship(candidate, objects, width, height):
    if not objects:
        return [np.nan, np.nan]
    b = np.asarray(candidate["box"])
    other = boxes([obj["box"] for obj in objects])
    centers = (other[:, :2] + other[:, 2:]) / 2
    center = (b[:2] + b[2:]) / 2
    distances = np.linalg.norm((centers - center) / [width, height], axis=1)
    nearest = int(distances.argmin())
    return [float(distances[nearest]), float(np.log((area(b) + 1) / (area(other[nearest]) + 1)))]


def image_features(image):
    """Low-cost image context, shared by B1/B2/M. No labels are read."""
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=float) / 255
    gradient = np.square(np.diff(gray, axis=0)).mean() + np.square(np.diff(gray, axis=1)).mean()
    return [float(gray.mean()), float(gray.std()), float(gradient)]


def evidence(predictions, visible_gt, cfg):
    """Only observed-class GT and predictions enter this function."""
    if not visible_gt:
        return np.full(15, np.nan)
    values = []
    for c in cfg["observed_classes"]:
        p = [p for p in predictions if p["cls"] == c and p["score"] >= cfg["evidence_conf"]]
        g = [g for g in visible_gt if g["cls"] == c]
        matched, overlap = match(p, g, cfg["correct_iou"])
        y = matched >= 0
        tp = int(y.sum())
        mean_iou = float(np.mean([overlap[i, j] for i, j in enumerate(matched) if j >= 0])) if tp else np.nan
        brier = float(np.mean([(obj["score"] - y[i]) ** 2 for i, obj in enumerate(p)])) if p else np.nan
        bias = float(np.mean([obj["score"] - y[i] for i, obj in enumerate(p)])) if p else np.nan
        values += [tp / len(p) if p else np.nan, tp / len(g) if g else np.nan, mean_iou, brier, bias]
    return np.asarray(values)


def candidate_features(candidate, predictions, visible_gt, width, height, context, cfg):
    """Returns nested feature sets B0/B1/B2; cannot access hidden GT."""
    x1, y1, x2, y2 = candidate["box"]
    w, h = x2 - x1, y2 - y1
    s = np.clip(candidate["score"], 1e-6, 1 - 1e-6)
    base = [s, np.log(s / (1 - s)), np.log(w / width), np.log(h / height),
            (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), np.log(w / h)]
    observed_pred = [p for p in predictions if p["cls"] in cfg["observed_classes"]]
    normal = list(context) + [np.log1p(len(predictions))]
    for c in cfg["observed_classes"]:
        scores = [p["score"] for p in observed_pred if p["cls"] == c]
        normal += [np.log1p(len(scores)), np.mean(scores) if scores else np.nan]
    normal += relationship(candidate, observed_pred, width, height)
    layout = [np.log1p(sum(g["cls"] == c for g in visible_gt)) for c in cfg["observed_classes"]]
    layout += relationship(candidate, visible_gt, width, height)
    return np.asarray(base), np.asarray(base + normal), np.asarray(base + normal + layout)


def shuffle_donors(image_records, scale, seed):
    """Matched image-level donors, excluding the same filename-prefix group."""
    rng = np.random.default_rng(seed)
    covariates = np.asarray([r["covariates"] for r in image_records]) / scale
    result = np.arange(len(image_records))
    distances = np.full(len(image_records), np.nan)
    for i, row in enumerate(image_records):
        if not row["has_evidence"]:
            continue
        eligible = [j for j, other in enumerate(image_records)
                    if other["has_evidence"] and other["group"] != row["group"]]
        if not eligible:
            raise ValueError("Shuffle needs evidence in at least two different groups.")
        d = np.linalg.norm(covariates[eligible] - covariates[i], axis=1)
        nearest = np.argsort(d, kind="stable")[:5]
        k = int(rng.choice(nearest))
        result[i], distances[i] = eligible[k], d[k]
    return result, distances
