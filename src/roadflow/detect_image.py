
from ultralytics import YOLO


def main():
    # Load a small, pretrained YOLO11 model.
    model = YOLO("yolo11n.pt")

    # Run detection on a sample image.
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        conf=0.25,
        save=True,
        project="runs/roadflow",
        name="first_detection",
        exist_ok=True,
        device="cpu",
    )

    # Display the detected objects.
    for result in results:
        print(f"Image: {result.path}")

        if result.boxes is not None:
            print(f"Objects detected: {len(result.boxes)}")

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = result.names[class_id]

                print(f"- {label}: {confidence:.2f}")


if __name__ == "__main__":
    main()
