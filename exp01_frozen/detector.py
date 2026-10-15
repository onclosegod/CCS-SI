"""Train or cache predictions using the existing local YOLO implementation."""
import argparse
import json
import os
from pathlib import Path
import sys

from common import HERE, NAMES, arguments, digest, read_jsonl, write_json


def main():
    parser = arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--weights", type=Path, help="Existing weights; provenance must be checked separately.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--roles", nargs="+", default=["quality_fit", "quality_val"], choices=["quality_fit", "quality_val", "test"])
    parser.add_argument("--limit", type=int, help="Smoke inference only; written to a separate cache.")
    args = parser.parse_args()
    out = args.run.resolve()
    cfg = json.loads((out / "config.json").read_text())
    os.environ.setdefault("YOLO_CONFIG_DIR", str(out / "yolo_settings"))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    detector_repo = str(Path(cfg["detector_repo"]).expanduser().resolve())
    sys.path.insert(0, detector_repo)
    # DDP workers start fresh interpreters and inherit the environment, not sys.path.
    inherited_paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                       if p and p != detector_repo]
    os.environ["PYTHONPATH"] = os.pathsep.join([detector_repo, *inherited_paths])
    from ultralytics import YOLO
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable in this session. Run on the GPU host, or use --device cpu for smoke inference.")
    if args.action == "train":
        initial = args.weights.resolve() if args.weights else out / "detector_arch.yaml"
        if not initial.is_file():
            raise FileNotFoundError(initial)
        if (out / "detector").exists():
            raise SystemExit("detector/ already exists; inspect its logs before starting another training run.")
        record = {"manifest_sha256": digest(out / "manifest.jsonl"), "initial_model": str(initial),
                  "initial_model_sha256": digest(initial), "initialization": "existing_weights_unverified" if args.weights else "random_architecture",
                  "epochs": args.epochs, "batch": args.batch, "imgsz": cfg["imgsz"], "seed": cfg["seed"]}
        write_json(out / "train_record.json", record)
        YOLO(str(initial)).train(data=str(out / "detector.yaml"), epochs=args.epochs, imgsz=cfg["imgsz"],
                                batch=args.batch, device=args.device, workers=4, seed=cfg["seed"],
                                deterministic=True, pretrained=False, amp=False, cache=False,
                                project=str(out), name="detector", exist_ok=False, patience=25,
                                plots=False, optimizer="AdamW", mosaic=0.5, close_mosaic=10)
        return
    weights = (args.weights or out / "detector/weights/best.pt").resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"No detector weights: {weights}. Train first, or provide --weights.")
    model = YOLO(str(weights))
    actual = [model.names[i] for i in range(len(model.names))]
    if actual != NAMES:
        raise ValueError(f"Expected ten VisDrone classes in this order: {NAMES}; got {actual}")
    records = read_jsonl(out / "manifest.jsonl")
    cache = out / ("smoke_predictions" if args.limit else "predictions")
    cache.mkdir(exist_ok=True)
    metadata = {"weights": str(weights), "weights_sha256": digest(weights),
                "manifest_sha256": digest(out / "manifest.jsonl"),
                "imgsz": cfg["imgsz"], "conf": cfg["candidate_conf"], "nms_iou": cfg["nms_iou"],
                "max_det": cfg["max_det"], "class_names": actual, "limit": args.limit}
    meta_path = cache / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Cache protocol differs. Use a new run directory.")
    write_json(meta_path, metadata)
    for role in args.roles:
        destination = cache / f"{role}.jsonl"
        if destination.exists():
            raise ValueError(f"Cache exists: {destination}; use a new run or inspect/remove only this generated cache.")
        selected = [r for r in records if r["role"] == role]
        if args.limit:
            selected = selected[:args.limit]
        # JSONL is streamed to a temporary path; interrupted work cannot look complete.
        partial = destination.with_suffix(".jsonl.partial")
        with partial.open("w") as stream:
            results = model.predict(source=[r["image"] for r in selected], stream=True, batch=args.batch,
                                    imgsz=cfg["imgsz"], conf=cfg["candidate_conf"], iou=cfg["nms_iou"],
                                    max_det=cfg["max_det"], agnostic_nms=False, device=args.device,
                                    save=False, verbose=False)
            count = 0
            for record, result in zip(selected, results):
                if Path(result.path).resolve() != Path(record["image"]).resolve():
                    raise ValueError("Prediction/source image order mismatch.")
                xyxy = result.boxes.xyxy.cpu().numpy().tolist()
                scores = result.boxes.conf.cpu().numpy().tolist()
                classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
                predictions = [{"box": b, "score": s, "cls": c} for b, s, c in zip(xyxy, scores, classes)]
                stream.write(json.dumps({"id": record["id"], "predictions": predictions}) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"{role}: {count}/{len(selected)}", flush=True)
        if count != len(selected):
            raise ValueError("Inference ended before every image was processed.")
        partial.rename(destination)
        print(f"Saved {destination}: {count} images", flush=True)


if __name__ == "__main__":
    main()
