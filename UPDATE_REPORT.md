# Updated Fix & Test Report

## Status

This package is rebuilt from the uploaded project after recovering its source files from a structurally incomplete ZIP archive. **No original project credentials, Git metadata, or historical PDF reports are included in the rebuilt package.**

## Changes implemented

### Detection coverage
- Primary detector remains the bundled Open Images V7 YOLO nano model.
- Optional supplemental COCO YOLO nano model is enabled by default to add common everyday-object coverage and downloads through Ultralytics if missing.
- Raw detections are retained before tracking/session confirmation.
- Press `D` in the camera preview to toggle orange raw boxes alongside green confirmed boxes.
- `diagnose.py` now reports raw detection counts/confidence separately from tracked IDs and confirmed sessions.
- `python diagnose.py --list-classes` prints the loaded primary and supplemental model class names.
- Phone/small-object confidence rules are lower than generic object rules.
- Body-part classes receive relaxed confirmation rules.

### Body parts and hands
- Optional YOLO pose module downloads `yolov8n-pose.pt` on first use, draws skeleton/keypoints, and sends keypoint observations (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) through the existing tracking/session pipeline.
- Optional MediaPipe hand-landmark module draws hand landmarks and adds left/right hand boxes to the existing sessions.
- MediaPipe is optional and installed via `requirements-hands.txt`; the main application continues if it is unavailable.

### Reliability and performance
- GUI-capable `opencv-python` replaces the headless package for `cv2.imshow` support.
- Model, tracker, scene classifier, and report paths are project-root based.
- Auto device selection supports CUDA, Apple MPS, or CPU fallback.
- MongoDB reconnect uses exponential backoff rather than attempting a connection on every frame.
- Detection, scene, and relationship writes are buffered in memory while MongoDB is offline and retried. Pending records are queryable and can be included in the current-run PDF. The buffer is capped at 5,000 records and is not persistent across a process crash.
- Live-state updates are throttled to once per second.
- Dashboard Live fragment refreshes every five seconds and shows device, raw detections, pose/hand status, and pending DB writes.
- Duplicate person boxes retain the higher-confidence detection.
- Object and relationship sessions close based on elapsed-time grace plus a small missed-observation count, reducing FPS dependence.
- Scene sessions close after an uncertainty/observation gap rather than bridging long gaps.
- PDF output is written under the project `results` folder by default.
- Windows launch script uses its own directory instead of a hard-coded user path. `setup.bat` creates a venv and installs dependencies.

### Security/distribution
- Removed `.env`, `.git`, and generated historical PDFs from the distribution.
- Added `.env.example` and strengthened `.gitignore`.
- Historical credentials were present in the originally uploaded archive; rotate MongoDB credentials and admin password if they are active.

## Test results

- Python syntax compilation: **passed**
- Unit tests: **25 passed**
- Session behavior: **passed** (ID reassociation, confirmation, low-confidence phone, flicker rejection, session durations)
- Expanded detection tests: **passed** (body-part confirmation rules, phone thresholds, class synonyms, supplemental duplicate merging)
- Scene gap test: **passed**
- Relationship time-based closure test: **passed**
- Offline DB queue test: **passed** (buffer and flush using mocked collection)
- PDF tests: **passed** (run scoping, duplicate removal, empty report handling; skips text extraction if `pdftotext` is unavailable)
- Primary Open Images weight file: **validated as a readable PyTorch ZIP container**
- Rebuilt distribution ZIP: **validated after creation**

## Not verified in this environment

This environment did not have Ultralytics, PyMongo, Streamlit, MediaPipe, or a webcam available. Therefore:
- Actual YOLO checkpoint loading and class-name extraction were not run.
- Real webcam inference/FPS was not measured.
- Phone/hand/body-part detection quality on your camera was not measured.
- COCO and pose model first-download behavior depends on internet access on your computer.
- MediaPipe hands requires installing the optional package on a supported Python version.

The code-level tests pass, but a final camera validation on your computer remains necessary. Use `python diagnose.py --seconds 60 --show --raw` and review raw detections before changing thresholds or training a custom model.
