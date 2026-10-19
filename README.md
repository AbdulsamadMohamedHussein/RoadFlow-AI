# RoadFlow AI 🚦

**An AI-powered traffic analytics and forecasting platform for smart-city applications.**

[![Project Status](https://img.shields.io/badge/status-early%20development-orange)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/AI-Computer%20Vision-purple)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

## 📌 Overview

RoadFlow AI is a long-term machine learning engineering project focused on extracting useful traffic information from road videos and developing traffic-volume forecasting models.

The platform will combine computer vision, data processing, machine learning, backend engineering, and a web dashboard. Development will begin with recorded traffic videos and a reproducible local prototype before progressing toward a more complete application.

This project builds on previous YOLO-based vehicle-detection work and will document the engineering process, experiments, failures, improvements, and verified results.

## 🎯 Project Objectives

- Detect vehicles in recorded road videos.
- Track vehicles across video frames.
- Count vehicles crossing a defined virtual line.
- Generate timestamped traffic-count datasets.
- Develop and evaluate traffic-volume forecasting models.
- Compare machine learning models against simple forecasting baselines.
- Build an API using FastAPI.
- Develop a React dashboard for displaying traffic analytics.
- Apply testing, version control, reproducibility, and responsible model evaluation.

## 🏙️ Intended Use Cases

Potential applications include:

- Traffic-volume analysis.
- Smart-city research and experimentation.
- Historical traffic-pattern analysis.
- Short-term traffic-volume forecasting.
- Educational demonstrations of applied machine learning.

**Important:** Vehicle volume is not the same as congestion. Reliable congestion estimation would require additional evidence and validation.

## 🧠 Planned System Architecture

The planned data flow is:

1. **Video input** — load a supported traffic video.
2. **Vehicle detection** — identify vehicles using a computer vision model.
3. **Vehicle tracking** — associate detections across video frames.
4. **Traffic counting** — count valid line crossings.
5. **Data processing** — aggregate counts into time intervals.
6. **Forecasting** — predict future traffic volume from historical observations.
7. **Backend API** — expose validated data and predictions.
8. **Web dashboard** — display counts, trends, forecasts, and evaluation results.

The architecture will evolve as the project is tested. Planned features will not be described as completed until they have been implemented and verified.

## 🛠️ Planned Technology Stack

| Area | Planned technologies |
|---|---|
| Programming | Python |
| Computer vision | OpenCV, YOLO |
| Vehicle tracking | ByteTrack or another evaluated tracker |
| Data analysis | NumPy, pandas, Matplotlib |
| Machine learning | scikit-learn; additional libraries if justified |
| Backend | FastAPI, Pydantic |
| Database | PostgreSQL |
| Frontend | React, JavaScript, charting libraries |
| Testing | pytest |
| Version control | Git and GitHub |
| Deployment | To be selected after the local prototype works |

## 📊 Model Evaluation

Model quality will be measured rather than assumed.

- **Detection:** precision, recall, and appropriate detection metrics.
- **Counting:** compare automated counts with manually verified counts.
- **Forecasting:** MAE and RMSE, evaluated against simple baselines.
- **Reliability:** record errors, processing time, and reproducibility.
- **Generalization:** test performance on data not used during training.

Forecasting experiments will use chronological data splits to reduce temporal leakage. Dataset limitations will be documented.

## 🗂️ Repository Structure

```text
RoadFlow-AI/
├── README.md
├── .gitignore
├── CHANGELOG.md
├── docs/
│   ├── project-charter.md
│   └── roadmap-700-days.md
└── journal/
    └── daily/
        └── day-001.md
```

Additional directories for source code, tests, datasets, models, experiments, and reports will be introduced as they become necessary.

Large datasets, credentials, and generated model weights will not be committed to GitHub without a specific reason.

## 🗓️ Development Journey

The project is organized around a long-term engineering journey, with regular documentation of:

- Daily learning and implementation.
- Weekly progress and blockers.
- Dataset preparation and experiments.
- Model evaluation and error analysis.
- Software architecture decisions.
- Testing, deployment, and improvements.
- Lessons learned and portfolio preparation.

The original planning window is October 15, 2026, to September 15, 2028. Actual progress and milestone dates will be recorded honestly.

## 📍 Current Status

**Phase:** Project setup and existing-work audit.

Current priorities:

- Establish repository documentation.
- Review the existing RoadFlow Detection project.
- Verify available datasets and trained model weights.
- Reproduce a vehicle-detection result.
- Define a reliable process for converting detections into traffic counts.

## Current Implementation

- `src/roadflow/detect_image.py` — single-image object detection.
- `src/roadflow/detect_video.py` — frame-by-frame video detection and annotation.
- `journal/daily/day-001.md` through `day-004.md` — daily development records.
- `data/raw/` — local input videos; excluded from Git.
- `runs/roadflow/` — local experiment outputs; excluded from Git.

The current video pipeline uses a pretrained YOLO11n model and OpenCV.
Video detection has been tested on a small sample, but road-vehicle
detection, object tracking, counting, and forecasting remain future work.


## 👨‍💻 Author

**Abdulsamad Mohamed**

Bachelor's student in Artificial Intelligence | Aspiring Machine Learning Engineer

## ⚖️ Data, Ethics, and Limitations

Datasets and video sources will be checked for licensing and permitted use. The project will prioritize traffic-level analysis rather than identifying individuals. Dataset bias, counting errors, environmental limitations, and model uncertainty will be documented.

## 📜 License

A license will be selected before public reuse of the project's code is encouraged. Until then, please check the repository for an explicit license before reusing its contents.

---

*RoadFlow AI is a work in progress. Claims about accuracy, performance, and completed functionality will be supported by recorded experiments and verifiable evidence.*