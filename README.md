# LCM_AI — detecting foreign cans on the conveyor belt (YOLO + Django)

Developed a high-precision computer vision application using **YOLO** (Ultralytics) to automate the detection of **foreign cans** on production conveyors, preventing batch contamination from previous production runs. The system utilizes a **Django-based architecture** to process live camera feeds, applying **Region of Interest (ROI)** filtering with **polygon masking** to analyze only the conveyor region and reduce false positives from adjacent lines. A **REST API** synchronizes detection results with a centralized **SQLite** database and supports interactive workflows through a multi-step wizard UI.

Technologies: **Django**, **Python**, **SQLite**, **OpenCV**, **Ultralytics YOLO**, **Channels/WebSockets** (live streams / telemetry).

---

## What this repository contains

- **Multi-step wizard** for camera setup, background calibration, dataset collection, training, and live deployment.
- **Automatic labeling** based on differential foreground detection vs calibrated background (within ROI).
- **Manual / semi-automatic labeling flows** for dataset creation.
- **Training orchestration** around Ultralytics YOLO checkpoints (`best.pt`).
- **Live detection loop** streaming annotated frames over WebSockets.

> Note: External annotation workflows (e.g. Label Studio) can be used alongside exported YOLO datasets; this repo focuses on an integrated factory workflow.

---

## Requirements

- Python **3.12+** recommended (project has been run on **Python 3.14** in development environments).
- Dependencies are listed in `requirements.txt`.

Optional:

- **CUDA** GPU for faster training/inference (falls back to CPU).

---

## Quick start (local development)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser  # optional
python manage.py runserver 0.0.0.0:8000
```

For WebSockets / ASGI (recommended if you use streaming endpoints):

```bash
daphne -b 0.0.0.0 -p 8000 sentinel_ai.asgi:application
```

---

## Configuration notes

- Camera streams are typically **MJPEG** over HTTP (Axis-compatible URLs supported).
- ROI can be stored as a **polygon** (normalized coordinates) to ignore neighboring conveyors.
- Large artifacts (weights, runs, datasets) are intentionally ignored by `.gitignore` — do not commit them.

---

## Repository hygiene

This repo includes a `.gitignore` tuned for Django + ML workflows:

- virtual environments (`venv/`, `.venv/`)
- SQLite databases (`*.sqlite3`, journals)
- datasets / weights / training artifacts (`datasets/`, `*.pt`, `runs/`, etc.)
- secrets (`.env`, keys)

---

## License / attribution

Project materials created for industrial conveyor QA use-cases; adapt licensing as needed for your organization.
