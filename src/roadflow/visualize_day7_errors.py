
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "runs" / "roadflow" / "experiments" / "day7_frames"
LABEL_DIR = ROOT / "data" / "labels" / "day7_eval"
MODEL_PATH = ROOT / "yolo11n.pt"
OUTPUT_DIR = ROOT / "runs" / "roadflow" / "experiments" / "day7_visual_review"

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50

# Manual class ID: (YOLO COCO class ID, class name)
CLASS_MAP = {
    0: (2, "car"),
    1: (7, "truck"),
}

GT_GREEN = (0, 200, 0)
PRED_CYAN = (255, 220, 0)
FP_RED = (0, 0, 255)
FN_YELLOW = (0, 220, 255)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def draw_box(image, box, color, label):
    x1, y1, x2, y2 = [round(v) for v in box]
    h, w = image.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(14, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def load_gt(path, width, height):
    ground_truth = {0: [], 1: []}

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label in {path.name}: {line}")

        class_id = int(parts[0])
        if class_id not in CLASS_MAP:
            continue

        xc, yc, bw, bh = map(float, parts[1:])
        box = [
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ]
        ground_truth[class_id].append(box)

    return ground_truth


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError("Image or label directory is missing.")

    label_files = sorted(LABEL_DIR.glob("frame_*.txt"))
    if not label_files:
        raise FileNotFoundError("No Day 7 label files found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    for label_path in label_files:
        image_path = IMAGE_DIR / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]
        gt = load_gt(label_path, width, height)

        result = model.predict(
            source=image,
            conf=CONFIDENCE,
            device="cpu",
            classes=[2, 7],
            verbose=False,
        )[0]

        predictions = {0: [], 1: []}

        if result.boxes is not None:
            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                for manual_id, (model_id, name) in CLASS_MAP.items():
                    if int(cls) == model_id:
                        predictions[manual_id].append(
                            (box, float(conf))
                        )
                        break

        canvas = image.copy()
        matched_count = 0
        fp_count = 0
        fn_count = 0

        for manual_id, (_, name) in CLASS_MAP.items():
            gt_boxes = gt[manual_id]
            pred_boxes = predictions[manual_id]

            candidates = []
            for pi, (pred_box, _) in enumerate(pred_boxes):
                for gi, gt_box in enumerate(gt_boxes):
                    overlap = iou(pred_box, gt_box)
                    if overlap >= IOU_THRESHOLD:
                        candidates.append((overlap, pi, gi))

            candidates.sort(reverse=True)
            used_predictions = set()
            used_gt = set()
            matches = {}

            for overlap, pi, gi in candidates:
                if pi not in used_predictions and gi not in used_gt:
                    used_predictions.add(pi)
                    used_gt.add(gi)
                    matches[pi] = overlap

            # Draw ground truth boxes.
            for gi, gt_box in enumerate(gt_boxes):
                if gi in used_gt:
                    draw_box(
                        canvas, gt_box, GT_GREEN,
                        f"GT {name} matched"
                    )
                    matched_count += 1
                else:
                    draw_box(
                        canvas, gt_box, FN_YELLOW,
                        f"MISSED GT {name}"
                    )
                    fn_count += 1

            # Draw model predictions.
            for pi, (pred_box, confidence) in enumerate(pred_boxes):
                if pi in used_predictions:
                    draw_box(
                        canvas, pred_box, PRED_CYAN,
                        f"YOLO {name} {confidence:.2f}"
                    )
                else:
                    draw_box(
                        canvas, pred_box, FP_RED,
                        f"UNMATCHED {name} {confidence:.2f}"
                    )
                    fp_count += 1

        legend = (
            f"{label_path.stem} | Matched: {matched_count} | "
            f"Unmatched predictions: {fp_count} | "
            f"Missed labels: {fn_count}"
        )

        cv2.rectangle(canvas, (0, 0), (width, 25), (30, 30, 30), -1)
        cv2.putText(
            canvas, legend, (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        output_path = OUTPUT_DIR / f"{label_path.stem}_review.jpg"
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"Could not save {output_path}")

        print(f"Saved: {output_path.name}")

    print(f"\nVisual review complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
