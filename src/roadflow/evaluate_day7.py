
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "runs" / "roadflow" / "experiments" / "day7_frames"
LABEL_DIR = PROJECT_ROOT / "data" / "labels" / "day7_eval"
MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50

# Manual label ID -> YOLO COCO class ID
CLASS_MAP = {
    0: (2, "car"),
    1: (7, "truck"),
}


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(label_path, width, height):
    ground_truth = {0: [], 1: []}

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label row in {label_path.name}: {line}")

        manual_id = int(parts[0])
        if manual_id not in CLASS_MAP:
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        box = [
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ]

        ground_truth[manual_id].append(box)

    return ground_truth


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    label_files = sorted(LABEL_DIR.glob("frame_*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No labels found in {LABEL_DIR}")

    model = YOLO(str(MODEL_PATH))

    totals = {
        manual_id: {"tp": 0, "fp": 0, "fn": 0}
        for manual_id in CLASS_MAP
    }

    print(f"Evaluating {len(label_files)} labelled frames")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"IoU matching threshold: {IOU_THRESHOLD}")

    for label_path in label_files:
        image_path = IMAGE_DIR / f"{label_path.stem}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = model.predict(
            source=str(image_path),
            conf=CONFIDENCE,
            device="cpu",
            classes=[2, 7],
            verbose=False,
        )

        result = results[0]
        height, width = result.orig_shape

        gt = load_ground_truth(label_path, width, height)
        predictions = {0: [], 1: []}

        if result.boxes is not None:
            for box, class_value in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                model_id = int(class_value)

                for manual_id, (expected_model_id, _) in CLASS_MAP.items():
                    if model_id == expected_model_id:
                        predictions[manual_id].append(box)
                        break

        for manual_id in CLASS_MAP:
            gt_boxes = gt[manual_id]
            pred_boxes = predictions[manual_id]

            candidates = []

            for pred_index, pred_box in enumerate(pred_boxes):
                for gt_index, gt_box in enumerate(gt_boxes):
                    overlap = calculate_iou(pred_box, gt_box)

                    if overlap >= IOU_THRESHOLD:
                        candidates.append(
                            (overlap, pred_index, gt_index)
                        )

            # Match the highest-IoU pairs first, without reusing boxes.
            candidates.sort(reverse=True)

            matched_predictions = set()
            matched_ground_truth = set()

            for _, pred_index, gt_index in candidates:
                if (
                    pred_index not in matched_predictions
                    and gt_index not in matched_ground_truth
                ):
                    matched_predictions.add(pred_index)
                    matched_ground_truth.add(gt_index)

            totals[manual_id]["tp"] += len(matched_predictions)
            totals[manual_id]["fp"] += (
                len(pred_boxes) - len(matched_predictions)
            )
            totals[manual_id]["fn"] += (
                len(gt_boxes) - len(matched_ground_truth)
            )

        print(f"Evaluated {label_path.stem}")

    print("\n=== DAY 7 BASELINE RESULTS ===")

    for manual_id, (_, name) in CLASS_MAP.items():
        tp = totals[manual_id]["tp"]
        fp = totals[manual_id]["fp"]
        fn = totals[manual_id]["fn"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(f"\n{name.upper()}")
        print(f"True positives:  {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"Precision:       {precision:.3f}")
        print(f"Recall:          {recall:.3f}")
        print(f"F1 score:        {f1:.3f}")

    print("\nEvaluation complete. No model training was performed.")


if __name__ == "__main__":
    main()
