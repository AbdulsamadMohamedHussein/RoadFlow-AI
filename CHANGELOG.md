# Changelog

All notable changes to RoadFlow-AI will be documented here.

## 2026-10-17 — Day 1
- Established the project documentation and initial roadmap.
- Defined the project mission, scope, and long-term milestones.
- Created the initial GitHub repository structure.

## 2026-10-18 — Day 2
- Verified the local Git repository and GitHub remote.
- Confirmed that the main branch is synchronized with origin/main.
- Confirmed the working tree is clean.
- Identified the next step: inspect the existing RoadFlow_Detection project before integrating reusable code.

## Day 3 — October 18, 2026

### Added
- Created an isolated Python virtual environment.
- Added the initial image-detection program at `src/roadflow/detect_image.py`.
- Ran the first YOLO11n inference experiment.
- Saved and visually verified the annotated sample image.

### Verified
- Python 3.14.7 environment.
- Successful imports of Ultralytics, OpenCV, and PyTorch.
- Five detections on the sample image: one bus and four people.

### Notes
- Initial inference uses CPU-only PyTorch.
- Video processing, tracking, counting, and forecasting remain planned work.
