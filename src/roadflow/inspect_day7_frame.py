
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "runs" / "roadflow" / "experiments" / "day7_frames"
LABEL_DIR = ROOT / "data" / "labels" / "day7_eval"
MODEL_PATH = ROOT / "yolo11n.pt"
OUTPUT_PATH = ROOT / "runs" / "roadflow" / "experiments" / "day7_visual_review" / "frame_00_side_by_side.jpg"

FRAME = "frame_06"
CONFIDENCE = 0.25

# Manual class ID: (YOLO COCO ID, name)
CLASS_MAP = {
    0: (2, "car"),
    1: (7, "truck"),
}


def draw_box(image, box, color, label):
    x1, y1, x2, y2 = [round(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image, label, (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )


def main():
    image_path = IMAGE_DIR / f"{FRAME}.jpg"
    label_path = LABEL_DIR / f"{FRAME}.txt"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Labels not found: {label_path}")

    height, width = image.shape[:2]
    manual_view = image.copy()
    prediction_view = image.copy()

    # Draw manual annotations on the left.
    manual_count = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        if class_id not in CLASS_MAP:
            continue

        _, name = CLASS_MAP[class_id]
        xc, yc, bw, bh = map(float, parts[1:])
        box = [
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ]
        draw_box(manual_view, box, (0, 200, 0), name)
        manual_count += 1

    # Draw pretrained YOLO predictions on the right.
    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        source=image,
        conf=CONFIDENCE,
        device="cpu",
        classes=[2, 7],
        verbose=False,
    )[0]

    prediction_count = 0
    if result.boxes is not None:
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            model_id = int(cls)
            for _, (expected_id, name) in CLASS_MAP.items():
                if model_id == expected_id:
                    draw_box(
                        prediction_view, box, (0, 0, 255),
                        f"{name} {float(conf):.2f}"
                    )
                    prediction_count += 1
                    break

    # Add clear headings.
    for view, heading in (
        (manual_view, f"MANUAL LABELS: {manual_count}"),
        (prediction_view, f"YOLO PREDICTIONS: {prediction_count}"),
    ):
        cv2.rectangle(view, (0, 0), (width, 28), (35, 35, 35), -1)
        cv2.putText(
            view, heading, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    comparison = cv2.hconcat([manual_view, prediction_view])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), comparison):
        raise RuntimeError("Could not save comparison image.")

    print(f"Manual boxes: {manual_count}")
    print(f"YOLO predictions: {prediction_count}")
    print(f"Saved comparison: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
