# lcm_ai
LCM_AI Yolo and ML app

Developed a high-precision computer vision application using YOLO instance
segmentation to automate the detection of "foreign cans" on production
conveyors, preventing batch contamination from previous production runs.
The system utilizes a Django-based architecture to process live camera
feeds, applying custom Region of Interest (ROI) filtering and polygonal
masking to identify and count any cans left on the line. By integrating a REST
API for real-time telemetry, the solution synchronizes detection data with a
centralized database, replacing manual visual inspections with a robust,
automated deep learning pipeline. The project involved the full ML lifecycle,
including dataset annotation in Label Studio and specialized transfer learning
with layer freezing to ensure high detection accuracy in dense industrial
environments.
(Technologies: Django, AI, ML, SQLite, Python)
