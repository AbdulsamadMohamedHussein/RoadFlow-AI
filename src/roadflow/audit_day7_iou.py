
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "runs/roadflow/experiments/day7_frames/frame_00.jpg"
LABELS = ROOT / "data/labels/day7_eval/frame_00.txt"
MODEL = ROOT / "yolo11n.pt"
OUTPUT = (
    ROOT / "runs/roadflow/experiments/day7_visual_review"
    / "frame_00_iou_audit.jpg"
)

CLASS_MAP = {0: (2, "car"), 1: (7, "truck")}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def draw_box(image, box, color, label):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(round, box)
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image, label, (x1, max(12, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA
    )


def main():
    image = cv2.imread(str(IMAGE))
    if image is None:
        raise FileNotFoundError(f"Image missing: {IMAGE}")

    if not LABELS.is_file():
        raise FileNotFoundError(f"Labels missing: {LABELS}")

    h, w = image.shape[:2]
    gt = {0: [], 1: []}

    for line in LABELS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label row: {line}")

        cls = int(parts[0])
        if cls not in CLASS_MAP:
            continue

        xc, yc, bw, bh = map(float, parts[1:])
        box = [
            (xc - bw / 2) * w,
            (yc - bh / 2) * h,
            (xc + bw / 2) * w,
            (yc + bh / 2) * h,
        ]
        gt[cls].append(box)

    result = YOLO(str(MODEL)).predict(
        source=image,
        conf=0.25,
        device="cpu",
        classes=[2, 7],
        verbose=False,
    )[0]

    pred = {0: [], 1: []}

    if result.boxes is not None:
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            for manual_cls, (model_cls, _) in CLASS_MAP.items():
                if int(cls) == model_cls:
                    pred[manual_cls].append((box, float(conf)))
                    break

    # Independent best-overlap diagnostics, not one-to-one matching.
    left = image.copy()
    right = image.copy()

    for cls, (_, name) in CLASS_MAP.items():
        for i, gt_box in enumerate(gt[cls], 1):
            overlaps = [iou(gt_box, pbox) for pbox, _ in pred[cls]]
            best = max(overlaps, default=0.0)

            if best >= 0.50:
                color = (0, 200, 0)
            elif best >= 0.30:
                color = (0, 200, 255)
            else:
                color = (0, 0, 255)

            draw_box(left, gt_box, color, f"GT {name} {i}: IoU {best:.2f}")

        for i, (pbox, confidence) in enumerate(pred[cls], 1):
            overlaps = [iou(pbox, gbox) for gbox in gt[cls]]
            best = max(overlaps, default=0.0)

            draw_box(
                right, pbox, (0, 0, 255),
                f"P {name} {i}: IoU {best:.2f}, conf {confidence:.2f}"
            )

    for panel, heading in [
        (left, "MANUAL BOXES: colour shows best IoU"),
        (right, "YOLO PREDICTIONS: red boxes"),
    ]:
        cv2.rectangle(panel, (0, 0), (w, 25), (30, 30, 30), -1)
        cv2.putText(
            panel, heading, (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    comparison = cv2.hconcat([left, right])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT), comparison):
        raise RuntimeError("Could not save the audit image.")

    print("Green: best IoU >= 0.50")
    print("Yellow: best IoU >= 0.30 and < 0.50")
    print("Red manual boxes: best IoU < 0.30")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
