
from pathlib import Path

import cv2
from ultralytics import YOLO


# Locate the RoadFlow-AI project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input, model, and output paths.
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "traffic.mp4"
MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "roadflow" / "video_detection"
OUTPUT_PATH = OUTPUT_DIR / "traffic_annotated.mp4"


def main():
    # Check that the required files exist.
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    # Open the input video.
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    writer = None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("The video has invalid metadata.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Prepare the output video with the same size and frame rate.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(OUTPUT_PATH),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create output video: {OUTPUT_PATH}"
            )

        # Load our existing pretrained model.
        model = YOLO(str(MODEL_PATH))

        frame_count = 0
        detection_count = 0

        print(f"Input video: {VIDEO_PATH}")
        print(f"Resolution: {width}x{height} at {fps:.2f} FPS")
        print("Starting CPU inference...")

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Detect objects in this frame.
            result = model.predict(
                source=frame,
                conf=0.25,
                device="cpu",
                verbose=False,
            )[0]

            # Draw detection boxes and labels.
            annotated_frame = result.plot()

            # Write the annotated frame to the output video.
            writer.write(annotated_frame)

            frame_count += 1

            if result.boxes is not None:
                detection_count += len(result.boxes)

            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")

        if frame_count == 0:
            raise RuntimeError("No video frames were processed.")

        print("\nVideo detection completed!")
        print(f"Frames processed: {frame_count}")
        print(f"Detections across frames: {detection_count}")
        print(f"Output saved to: {OUTPUT_PATH}")

    finally:
        cap.release()

        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
