from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "runs" / "roadflow" / "experiments" / "day7_frames"
LABEL_DIR = PROJECT_ROOT / "data" / "labels" / "day7_eval"

CLASSES = {
    0: ("car", (0, 255, 0)),
    1: ("truck", (0, 200, 255)),
    2: ("bus", (255, 0, 255)),
    3: ("motorcycle", (255, 180, 0)),
}

KEY_CLASSES = {
    ord("c"): 0,
    ord("t"): 1,
    ord("b"): 2,
    ord("m"): 3,
}

images = sorted(IMAGE_DIR.glob("frame_*.jpg"))

if not images:
    raise FileNotFoundError(f"No frame images found in {IMAGE_DIR}")

LABEL_DIR.mkdir(parents=True, exist_ok=True)

index = 6
selected_class = 0
boxes = []
drag_start = None
drag_end = None
current_image = None


def label_path():
    return LABEL_DIR / f"{images[index].stem}.txt"


def load_labels():
    result = []

    path = label_path()
    if not path.exists():
        return result

    height, width = current_image.shape[:2]

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        x1 = round((xc - bw / 2) * width)
        y1 = round((yc - bh / 2) * height)
        x2 = round((xc + bw / 2) * width)
        y2 = round((yc + bh / 2) * height)

        if class_id in CLASSES:
            result.append((class_id, x1, y1, x2, y2))

    return result


def save_labels():
    height, width = current_image.shape[:2]
    lines = []

    for class_id, x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        lines.append(
            f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    label_path().write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    print(f"Saved {len(boxes)} boxes: {label_path().relative_to(PROJECT_ROOT)}")


def render():
    display = current_image.copy()

    for class_id, x1, y1, x2, y2 in boxes:
        name, color = CLASSES[class_id]
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display, name, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    if drag_start is not None and drag_end is not None:
        cv2.rectangle(display, drag_start, drag_end, (255, 255, 255), 1)

    class_name = CLASSES[selected_class][0]
    cv2.putText(
        display,
        f"{images[index].name} | Class: {class_name} | Boxes: {len(boxes)}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2,
    )
    cv2.putText(
        display,
        "C car | T truck | B bus | M motorcycle | U undo | N/P navigate | Q quit",
        (8, display.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1,
    )

    cv2.imshow("RoadFlow AI - Manual Labeler", display)


def mouse_event(event, x, y, flags, param):
    global drag_start, drag_end, boxes

    height, width = current_image.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        drag_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
        drag_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
        x1 = min(drag_start[0], x)
        y1 = min(drag_start[1], y)
        x2 = max(drag_start[0], x)
        y2 = max(drag_start[1], y)

        if x2 - x1 >= 4 and y2 - y1 >= 4:
            boxes.append((selected_class, x1, y1, x2, y2))

        drag_start = None
        drag_end = None


cv2.namedWindow("RoadFlow AI - Manual Labeler", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("RoadFlow AI - Manual Labeler", mouse_event)

try:
    while True:
        image_path = images[index]
        current_image = cv2.imread(str(image_path))

        if current_image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        boxes = load_labels()
        render()

        print(f"\nImage {index + 1}/{len(images)}: {image_path.name}")
        print("Draw boxes with the left mouse button.")
        print("Choose class with C/T/B/M. Press N when finished.")

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in KEY_CLASSES:
                selected_class = KEY_CLASSES[key]
                render()

            elif key == ord("u"):
                if boxes:
                    boxes.pop()
                render()

            elif key == ord("n"):
                save_labels()
                index = (index + 1) % len(images)
                break

            elif key == ord("p"):
                save_labels()
                index = (index - 1) % len(images)
                break

            elif key == ord("q"):
                save_labels()
                raise SystemExit

finally:
    cv2.destroyAllWindows()

