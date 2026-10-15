"""Fit nested reliability baselines on quality_fit, select on quality_val, test once."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from common import (arguments, candidate_features, digest, evidence, ignore_filter,
                    image_features, ious, match, read_jsonl, shuffle_donors, write_json)

METHODS = ("B0", "B1", "B2", "M", "M_shuffle")


def build_role(out, role, cfg):
    cache = out / "predictions" / f"{role}.jsonl"
    signature = {"manifest": digest(out / "manifest.jsonl"), "predictions": digest(cache),
                 "config": digest(out / "config.json"), "common_code": digest(Path(__file__).with_name("common.py"))}
    prefix = out / f"features_{role}"
    meta_path = prefix.with_suffix(".json")
    array_path = prefix.with_suffix(".npz")
    if meta_path.exists() and array_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["signature"] != signature:
            raise ValueError(f"Stale features for {role}; use a new run or remove only this generated feature cache.")
        with np.load(array_path) as arrays:
            return dict(arrays), metadata["images"]
    records = [r for r in read_jsonl(out / "manifest.jsonl") if r["role"] == role]
    prediction_rows = read_jsonl(cache)
    predictions = {r["id"]: r["predictions"] for r in prediction_rows}
    if len(predictions) != len(prediction_rows) or set(predictions) != {r["id"] for r in records}:
        raise ValueError(f"{role}: prediction IDs must match the complete manifest role exactly.")
    rows = {k: [] for k in ("B0", "B1", "B2", "y", "score", "image_index", "box")}
    image_rows, evidence_rows = [], []
    for image_index, record in enumerate(records):
        width, height = record["width"], record["height"]
        gt = ignore_filter(record["gt"], record["ignore"], width, height, cfg["ignore_ioa"])
        pred = ignore_filter(predictions[record["id"]], record["ignore"], width, height, cfg["ignore_ioa"])
        visible = [g for g in gt if g["cls"] in cfg["observed_classes"]]
        hidden_target = [g for g in gt if g["cls"] == cfg["target_class"]]
        candidates = [p for p in pred if p["cls"] == cfg["target_class"]]
        with Image.open(record["image"]) as image:
            context = image_features(image)
        evidence_rows.append(evidence(pred, visible, cfg))
        # All shuffle matching variables are available without hidden labels.
        visible_areas = [(g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1]) / (width * height) for g in visible]
        covariates = [np.log1p(len(visible)), np.log(np.mean(visible_areas) + 1e-8) if visible else 0,
                      context[0], context[1], np.log1p(len(pred))]
        image_rows.append({"id": record["id"], "group": record["group"], "target_gt": hidden_target,
                           "has_evidence": bool(visible), "covariates": covariates,
                           "candidate_count": len(candidates), "raw_prediction_count": len(predictions[record["id"]])})
        assigned, _ = match(candidates, hidden_target, cfg["correct_iou"])
        for i, candidate in enumerate(candidates):
            features = candidate_features(candidate, pred, visible, width, height, context, cfg)
            for name, values in zip(("B0", "B1", "B2"), features):
                rows[name].append(values)
            rows["y"].append(int(assigned[i] >= 0))
            rows["score"].append(candidate["score"])
            rows["image_index"].append(image_index)
            rows["box"].append(candidate["box"])
        if (image_index + 1) % 200 == 0:
            print(f"Features {role}: {image_index + 1}/{len(records)}", flush=True)
    if not rows["y"]:
        raise ValueError(f"{role}: no target-class candidates. Check detector/class mapping/pool threshold.")
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays["image_index"] = arrays["image_index"].astype(int)
    arrays["E"] = np.asarray(evidence_rows)
    np.savez_compressed(array_path, **arrays)
    write_json(meta_path, {"signature": signature, "images": image_rows})
    return arrays, image_rows


def design_matrices(data, images, scale, seed):
    donors, distances = shuffle_donors(images, scale, seed)
    indices = data["image_index"]
    matrices = {k: data[k] for k in ("B0", "B1", "B2")}
    matrices["M"] = np.column_stack((data["B2"], data["E"][indices]))
    matrices["M_shuffle"] = np.column_stack((data["B2"], data["E"][donors[indices]]))
    audit = [{"image": row["id"], "donor": images[int(donors[i])]["id"] if row["has_evidence"] else None,
              "distance": float(distances[i]) if np.isfinite(distances[i]) else None}
             for i, row in enumerate(images)]
    return matrices, audit


def losses(y, probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), np.square(p - y)


def metrics(y, p):
    nll, brier = losses(y, p)
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "positive_rate": float(np.mean(y)), "nll": float(nll.mean()),
            "brier": float(brier.mean()), "auroc": float(roc_auc_score(y, p)) if both else None,
            "auprc": float(average_precision_score(y, p)) if both else None}


def selection_metrics(data, images, probabilities, cfg):
    """Rematch after selection: a previously duplicate box can become a TP."""
    target_count = sum(len(r["target_gt"]) for r in images)
    by_image = [np.flatnonzero(data["image_index"] == i) for i in range(len(images))]
    coverage = 0
    for row, idx in zip(images, by_image):
        overlap = ious(data["box"][idx], [g["box"] for g in row["target_gt"]])
        if len(idx) and row["target_gt"]:
            coverage += int((overlap.max(axis=0) >= cfg["correct_iou"]).sum())
    result = {"target_gt_count": target_count,
              "candidate_pool_gt_coverage_upper_bound": coverage / target_count if target_count else None}
    order = np.argsort(-probabilities, kind="stable")
    for fraction in cfg["selection_fractions"]:
        k = max(1, int(np.ceil(fraction * len(order))))
        chosen = np.zeros(len(order), dtype=bool)
        chosen[order[:k]] = True
        tp = 0
        for row, idx in zip(images, by_image):
            selected = idx[chosen[idx]]
            p = [{"cls": cfg["target_class"], "box": data["box"][i].tolist(), "score": float(probabilities[i])} for i in selected]
            assigned, _ = match(p, row["target_gt"], cfg["correct_iou"])
            tp += int((assigned >= 0).sum())
        result[str(fraction)] = {"selected": k, "tp": tp, "precision": tp / k,
                                 "recall": tp / target_count if target_count else None}
    return result


def bootstrap_delta(data, images, baseline, proposed, cfg):
    groups = np.asarray([images[i]["group"] for i in data["image_index"]])
    names, ids = np.unique(groups, return_inverse=True)
    counts = np.bincount(ids)
    rng = np.random.default_rng(cfg["seed"])
    draws = rng.integers(len(names), size=(cfg["bootstrap_repeats"], len(names)))
    result = {"resampling_unit": "filename_prefix_group", "groups": int(len(names)),
              "sign": "positive means M improves over B2"}
    for label, before, after in zip(("nll", "brier"), losses(data["y"], baseline), losses(data["y"], proposed)):
        delta = before - after
        summed = np.bincount(ids, weights=delta)
        samples = summed[draws].sum(axis=1) / counts[draws].sum(axis=1)
        result[label] = {"delta": float(delta.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    return result


def report(out, stage, data, images, scores, cfg, provenance):
    report_dir = out / f"results_{stage}"
    report_dir.mkdir(exist_ok=True)
    area = (data["box"][:, 2] - data["box"][:, 0]) * (data["box"][:, 3] - data["box"][:, 1])
    no_evidence = np.asarray([not images[i]["has_evidence"] for i in data["image_index"]])
    strata = {"all": np.ones(len(area), bool), "small_predicted_box": area < 32 ** 2,
              "medium_predicted_box": (area >= 32 ** 2) & (area < 96 ** 2),
              "large_predicted_box": area >= 96 ** 2, "no_visible_gt": no_evidence,
              "visible_gt_available": ~no_evidence}
    rows = []
    for method, p in scores.items():
        for name, mask in strata.items():
            if mask.any():
                rows.append({"method": method, "stratum": name, **metrics(data["y"][mask], p[mask])})
    with (report_dir / "quality_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(report_dir / "selection_metrics.json", {m: selection_metrics(data, images, p, cfg) for m, p in scores.items()})
    delta = bootstrap_delta(data, images, scores["B2"], scores["M"], cfg)
    write_json(report_dir / "delta_B2_vs_M.json", delta)
    write_json(report_dir / "audit.json", {"stage": stage, "images": len(images), "candidates": len(data["y"]),
               "no_visible_gt_images": sum(not r["has_evidence"] for r in images),
               "prediction_cap_images": sum(r["raw_prediction_count"] >= cfg["max_det"] for r in images),
               "detector_provenance": provenance,
               "label_rule": "original-score greedy same-class IoU match; fixed for NLL/Brier/AUC; rematched for selection",
               "scale_rule": "predicted box area in original image pixels",
               "context_limit": "B1 uses photometric statistics and prediction context, not learned backbone features"})
    np.savez_compressed(report_dir / "candidate_scores.npz", **scores)
    print(json.dumps(delta, indent=2), flush=True)
    print(f"Reports: {report_dir}", flush=True)


def evaluate(out, stage):
    cfg = json.loads((out / "config.json").read_text())
    metadata = json.loads((out / "predictions/metadata.json").read_text())
    if metadata["manifest_sha256"] != digest(out / "manifest.jsonl") or metadata["limit"] is not None:
        raise ValueError("Prediction cache is not a complete cache for this manifest.")
    train_record = out / "train_record.json"
    provenance = "unverified existing weights; exploratory only"
    if train_record.exists():
        rec = json.loads(train_record.read_text())
        if (rec["manifest_sha256"] == metadata["manifest_sha256"]
                and Path(metadata["weights"]).resolve() == (out / "detector/weights/best.pt").resolve()
                and rec["initialization"] == "random_architecture"):
            provenance = "trained from scratch on this run's detector-only splits"
    models_path = out / "reliability_models.joblib"
    lock_path = out / "analysis_lock.json"
    signature = {"manifest_sha256": digest(out / "manifest.jsonl"), "config_sha256": digest(out / "config.json"),
                 "prediction_protocol": metadata, "common_code": digest(Path(__file__).with_name("common.py")),
                 "evaluation_code": digest(Path(__file__))}
    if stage == "validation":
        if lock_path.exists():
            raise ValueError("Validation already locked. Inspect its results or prepare a new run.")
        fit, fit_images = build_role(out, "quality_fit", cfg)
        val, val_images = build_role(out, "quality_val", cfg)
        if len(np.unique(fit["y"])) != 2:
            raise ValueError("quality_fit needs both correct and incorrect target candidates.")
        scale = np.asarray([r["covariates"] for r in fit_images]).std(axis=0)
        scale[scale < 1e-8] = 1
        fit_x, fit_audit = design_matrices(fit, fit_images, scale, cfg["seed"])
        val_x, val_audit = design_matrices(val, val_images, scale, cfg["seed"] + 1)
        write_json(out / "shuffle_quality_fit.json", fit_audit)
        write_json(out / "shuffle_quality_val.json", val_audit)
        models, choices, scores = {}, {}, {"raw_score": val["score"]}
        for method in METHODS:
            best = None
            for leaves in cfg["leaf_candidates"]:
                model = HistGradientBoostingClassifier(max_leaf_nodes=leaves, max_iter=cfg["max_iter"],
                            min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=1,
                            early_stopping=False, random_state=cfg["seed"])
                model.fit(fit_x[method], fit["y"])
                probability = model.predict_proba(val_x[method])[:, 1]
                criterion = metrics(val["y"], probability)["nll"]
                if best is None or criterion < best[0]:
                    best = (criterion, model, probability, leaves)
            _, models[method], scores[method], choices[method] = best
            print(f"Selected {method}: leaves={choices[method]}, validation NLL={best[0]:.6f}", flush=True)
        joblib.dump({"models": models, "shuffle_scale": scale}, models_path)
        write_json(lock_path, {"signature": signature, "models_sha256": digest(models_path), "selected_leaves": choices})
        data, images = val, val_images
    else:
        lock = json.loads(lock_path.read_text())
        if lock["signature"] != signature or lock["models_sha256"] != digest(models_path):
            raise ValueError("Protocol/model changed after validation was locked. Prepare a new run.")
        if (out / "results_test/audit.json").exists():
            raise ValueError("Test already evaluated. Read the saved report; do not tune on test.")
        bundle = joblib.load(models_path)
        data, images = build_role(out, "test", cfg)
        x, audit = design_matrices(data, images, bundle["shuffle_scale"], cfg["seed"] + 2)
        write_json(out / "shuffle_test.json", audit)
        scores = {"raw_score": data["score"]}
        scores.update({method: model.predict_proba(x[method])[:, 1] for method, model in bundle["models"].items()})
    # Predeclared fallback: no visible GT -> use the B2 result for both evidence methods.
    no_evidence = np.asarray([not images[i]["has_evidence"] for i in data["image_index"]])
    for method in ("M", "M_shuffle"):
        scores[method][no_evidence] = scores["B2"][no_evidence]
    report(out, stage, data, images, scores, cfg, provenance)


if __name__ == "__main__":
    parser = arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--stage", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        evaluate(args.run.resolve(), args.stage)
