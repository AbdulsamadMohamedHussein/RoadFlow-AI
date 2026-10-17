# RoadFlow AI — 700-Day Roadmap

**Planned start:** October 15, 2026  
**Target completion:** September 15, 2028

## Phase 1 — Foundation (Days 1–30)

- Establish repository structure and documentation.
- Inspect the existing RoadFlow Detection project.
- Document the hardware, dataset, and baseline configuration.
- Establish a reproducible Python environment.
- Identify baseline evaluation requirements.

**Milestone:** A documented and reproducible project foundation.

## Phase 2 — Detection (Days 31–120)

- Review the dataset and its licensing.
- Reproduce or validate the existing YOLO baseline.
- Implement evaluation scripts.
- Measure precision, recall, mAP, and inference latency where applicable.
- Investigate false positives and missed detections.
- Compare controlled model experiments.

**Milestone:** A documented detector evaluation with reproducible results.

## Phase 3 — Tracking (Days 121–240)

- Investigate tracking algorithms.
- Process sample road videos.
- Evaluate identity continuity and tracking failures.
- Explore vehicle counting using defined counting regions.
- Document tracking limitations.

**Milestone:** A tested tracking prototype, if feasible.

## Phase 4 — Traffic Analytics (Days 241–360)

- Define traffic indicators.
- Implement vehicle-count summaries.
- Explore traffic-flow measurements.
- Validate counting accuracy against manually checked samples.
- Produce plots and reports.

**Milestone:** An evaluated traffic analytics prototype.

## Phase 5 — Backend and Interface (Days 361–480)

- Define software architecture.
- Design API contracts.
- Develop FastAPI endpoints.
- Add input validation and automated tests.
- Build a dashboard for displaying available results.

**Milestone:** A working local application prototype, subject to feasibility.

## Phase 6 — Engineering and Deployment (Days 481–600)

- Profile inference performance.
- Improve logging and configuration.
- Add packaging and setup instructions.
- Test reproducibility on a clean environment.
- Investigate deployment options and resource costs.

**Milestone:** A documented, tested deployment candidate.

## Phase 7 — Final Evaluation and Release (Days 601–700)

- Run final evaluations.
- Compare results with the original baseline.
- Resolve important defects.
- Review security, licensing, and privacy.
- Prepare a final technical report.
- Record a demonstration.
- Publish a release with limitations and reproducibility instructions.

**Milestone:** A professional, evidence-backed portfolio release.

## Progress Tracking

Every phase should be supported by:

- Daily journal entries
- Weekly summaries
- Experiment records
- Git commits
- Issues and milestone reviews
- Technical reports and test results

## Review Rule

These are planning milestones, not guaranteed outcomes. Adjust the scope based on measured progress, hardware limitations, and university workload. Record important changes rather than silently rewriting history.
