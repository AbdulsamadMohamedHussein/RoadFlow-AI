
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "runs/roadflow/experiments/day7_frames"
LABEL_DIR = ROOT / "data/labels/day7_eval"
MODEL_PATH = ROOT / "yolo11n.pt"

CONFIDENCE = 0.25
THRESHOLDS = (0.50, 0.30)

# Manual class ID -> (YOLO COCO class ID, name)
CLASS_MAP = {
    0: (2, "car"),
    1: (7, "truck"),
}


def calculate_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def load_labels(path, width, height):
    boxes = {0: [], 1: []}

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label row in {path.name}: {line}")

        cls = int(parts[0])
        if cls not in CLASS_MAP:
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        if not (
            0 <= xc <= 1 and 0 <= yc <= 1
            and 0 < bw <= 1 and 0 < bh <= 1
        ):
            raise ValueError(f"Invalid normalized box in {path.name}: {line}")

        boxes[cls].append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ])

    return boxes


def match_boxes(gt, predictions, threshold):
    # Greedy one-to-one matching, same approach as the baseline.
    candidates = []

    for pi, pred in enumerate(predictions):
        for gi, truth in enumerate(gt):
            overlap = calculate_iou(pred, truth)
            if overlap >= threshold:
                candidates.append((overlap, pi, gi))

    candidates.sort(reverse=True)

    used_pred = set()
    used_gt = set()

    for _, pi, gi in candidates:
        if pi not in used_pred and gi not in used_gt:
            used_pred.add(pi)
            used_gt.add(gi)

    return len(used_pred), len(predictions) - len(used_pred), len(gt) - len(used_gt)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    labels = sorted(LABEL_DIR.glob("frame_*.txt"))
    if not labels:
        raise FileNotFoundError(f"No labels found in {LABEL_DIR}")

    totals = {
        cls: {
            threshold: {"tp": 0, "fp": 0, "fn": 0}
            for threshold in THRESHOLDS
        }
        for cls in CLASS_MAP
    }

    overlap_bins = {
        cls: {"below_030": 0, "from_030_to_050": 0, "at_least_050": 0}
        for cls in CLASS_MAP
    }

    model = YOLO(str(MODEL_PATH))

    print(f"Frames: {len(labels)}")
    print(f"Confidence: {CONFIDENCE}")
    print("Diagnostic IoU thresholds: 0.50 and 0.30")

    for label_path in labels:
        image_path = IMAGE_DIR / f"{label_path.stem}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        result = model.predict(
            source=str(image_path),
            conf=CONFIDENCE,
            device="cpu",
            classes=[2, 7],
            verbose=False,
        )[0]

        height, width = result.orig_shape
        gt = load_labels(label_path, width, height)
        predictions = {0: [], 1: []}

        if result.boxes is not None:
            for box, class_value in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                model_cls = int(class_value)

                for manual_cls, (expected_cls, _) in CLASS_MAP.items():
                    if model_cls == expected_cls:
                        predictions[manual_cls].append(box)
                        break

        print(f"\n{label_path.stem}")

        for cls, (_, name) in CLASS_MAP.items():
            gt_boxes = gt[cls]
            pred_boxes = predictions[cls]

            for truth in gt_boxes:
                best = max(
                    (calculate_iou(truth, pred) for pred in pred_boxes),
                    default=0.0,
                )

                if best >= 0.50:
                    overlap_bins[cls]["at_least_050"] += 1
                elif best >= 0.30:
                    overlap_bins[cls]["from_030_to_050"] += 1
                else:
                    overlap_bins[cls]["below_030"] += 1

            frame_results = {}

            for threshold in THRESHOLDS:
                tp, fp, fn = match_boxes(gt_boxes, pred_boxes, threshold)
                frame_results[threshold] = (tp, fp, fn)

                for key, value in zip(("tp", "fp", "fn"), (tp, fp, fn)):
                    totals[cls][threshold][key] += value

            tp50, fp50, fn50 = frame_results[0.50]
            tp30, fp30, fn30 = frame_results[0.30]

            print(
                f"  {name}: GT={len(gt_boxes)}, predictions={len(pred_boxes)} | "
                f"IoU .50 TP/FP/FN={tp50}/{fp50}/{fn50} | "
                f"IoU .30 TP/FP/FN={tp30}/{fp30}/{fn30}"
            )

    print("\n=== DATASET TOTALS ===")

    for cls, (_, name) in CLASS_MAP.items():
        print(f"\n{name.upper()}")

        for threshold in THRESHOLDS:
            stats = totals[cls][threshold]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            print(
                f"IoU {threshold:.2f}: TP={tp}, FP={fp}, FN={fn}, "
                f"Precision={precision:.3f}, Recall={recall:.3f}, "
                f"F1={f1:.3f}"
            )

        print("Manual boxes by best independent overlap:")
        print(f"  IoU < 0.30: {overlap_bins[cls]['below_030']}")
        print(f"  0.30 <= IoU < 0.50: {overlap_bins[cls]['from_030_to_050']}")
        print(f"  IoU >= 0.50: {overlap_bins[cls]['at_least_050']}")

    print("\nAudit complete. No labels or model weights were changed.")


if __name__ == "__main__":
    main()
