# AI-Based Real-Time Object Detection, Tracking, Scene Classification & Monitoring System

A real-time computer vision system that watches a live webcam feed and answers:

- **What objects are present?** (YOLO Open Images V7 (601 classes) + optional COCO supplement)
- **How long has each object been visible?** (per-object tracking sessions, unique IDs)
- **Which objects are visible together, and for how long?** (co-occurrence, not assumed relationships)
- **What kind of environment is this?** (Places365 scene classification — Room, Office, Street, etc.)
- **How long has that scene lasted?**
- **When did all of this happen?** (everything timestamped and stored in MongoDB)

All of the above is stored in MongoDB and viewable/searchable in a Streamlit dashboard.

```
LIVE WEBCAM → YOLO DETECTION → OBJECT TRACKING → SCENE CLASSIFIER → MONGODB → DASHBOARD
```

---

## 1. Project Structure

```
AI_Object_Detection_System/
│
├── main.py                 # Live camera loop: detection + tracking + scene + DB writes
├── detector.py              # Open Images + optional COCO YOLO detectors
├── tracker.py                # Turns per-frame detections into "detection sessions"
├── scene_classifier.py       # Places365 scene classifier + "scene sessions"
├── relationship.py           # Object co-occurrence session manager
├── database.py                # All MongoDB reads/writes (PyMongo), error-tolerant
├── config.py                  # All tunable settings in one place
├── dashboard.py               # Streamlit dashboard (history, stats, search)
├── download_places365.py      # One-time script to fetch scene-classifier weights
│
├── requirements.txt
├── README.md
│
├── models/
│   ├── object_detection_model/       # YOLO weights land here if you point YOLO_MODEL_PATH here
│   └── scene_classification_model/   # Places365 weights + categories file go here
│
├── templates/     # reserved (e.g. if you later add a Flask-based UI)
├── static/        # reserved (e.g. static assets for a Flask-based UI)
└── results/        # optional: exported snapshots/reports
```

---

## 2. Requirements

- Python 3.11 or 3.12 recommended (especially for optional MediaPipe hand support)
- A webcam (built-in or USB)
- MongoDB (local install **or** a free MongoDB Atlas cluster)
- ~2 GB free disk space (PyTorch + model weights)
- Works on CPU. A CUDA GPU (or Apple Silicon `mps`) makes it noticeably faster but isn't required.

---

## 3. Setup Instructions

### Step 1 — Get the code and create a virtual environment

```bash
cd AI_Object_Detection_System

python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Step 2 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs GUI-capable OpenCV, Ultralytics (YOLO), PyTorch/torchvision, PyMongo, Streamlit, pandas, numpy, and Pillow. Optional MediaPipe hands are installed separately. If you previously installed `opencv-python-headless`, remove it from the same environment.

> If `pip install torch` is slow or you want GPU support, install the CUDA build of PyTorch
> first using the selector at https://pytorch.org/get-started/locally/ , then run
> `pip install -r requirements.txt` again — it will skip torch/torchvision since they're already installed.

### Step 3 — Install and start MongoDB

**Option A — Local MongoDB (recommended for development)**

- macOS: `brew tap mongodb/brew && brew install mongodb-community && brew services start mongodb-community`
- Ubuntu/Debian: follow MongoDB's official apt instructions, then `sudo systemctl start mongod`
- Windows: install "MongoDB Community Server" from mongodb.com, it installs as a service and starts automatically

Verify it's running:

```bash
mongosh --eval "db.runCommand({ ping: 1 })"
```

**Option B — MongoDB Atlas (free cloud cluster, no local install)**

1. Create a free cluster at https://www.mongodb.com/cloud/atlas
2. Get your connection string (looks like `mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/`)
3. Set it as an environment variable before running the app:

```bash
# macOS/Linux
export MONGO_URI="mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/"
# Windows (PowerShell)
$env:MONGO_URI="mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/"
```

If `MONGO_URI` isn't set, the app defaults to `mongodb://localhost:27017` (see `config.py`).

The database (`object_detection_db`) and its collections (`detections`, `scene_sessions`,
`object_relationships`, `live_state`) are created automatically the first time the app writes to them —
you don't need to create them manually.

### Step 4 — Models

The Open Images V7 primary weight `yolov8n-oiv7.pt` is included. On first startup,
Ultralytics may download the optional COCO supplement (`yolov8n.pt`) and body-pose
model (`yolov8n-pose.pt`). An internet connection is needed for those first downloads.
If the optional models cannot download, the application logs a warning and continues
with the primary detector.

### Step 5 — Download the Places365 scene-classification weights (one-time)

```bash
python download_places365.py
```

This downloads `resnet18_places365.pth.tar` (~45 MB) and `categories_places365.txt` into
`models/scene_classification_model/`. If you skip this step, the app still runs — it just won't
show a scene prediction until you run it.

### Step 6 — Run the live camera app

```bash
python main.py
```

A window titled **"REAL-TIME AI CAMERA"** opens showing your webcam feed with bounding boxes,
object labels, confidence scores, tracking IDs, the current scene, currently co-occurring
objects, and FPS.

**Controls (click the video window first so it has focus):**

| Key | Action |
|-----|--------|
| `q` or `Esc` | Stop the camera and exit safely (flushes any open sessions to MongoDB) |
| `d` | Toggle raw detector boxes (orange) in addition to confirmed boxes (green) |
| `c` | Clear all history (asks for `yes`/`no` confirmation in the terminal) |
| `Ctrl+C` (in terminal) | Also stops safely |

Raw diagnostics:
```bash
python diagnose.py --seconds 60 --show --raw
python diagnose.py --list-classes
```

### Step 7 — Run the dashboard

In a **second terminal** (keep `main.py` running in the first one):

```bash
source .venv/bin/activate      # activate the same venv
streamlit run dashboard.py
```

This opens a browser tab with tabs for: **Live** status, **Detection History**, **Scene History**,
**Relationships**, **Search**, and **Admin** (clear history).

---

## 4. Configuration

Everything tunable lives in `config.py`:

| Setting | What it controls |
|---|---|
| `CAMERA_INDEX` | Which camera to use (0 = default); can also be a video file path for testing |
| `YOLO_MODEL_PATH` | Primary checkpoint (default: bundled Open Images V7 model) |
| `YOLO_CONFIDENCE_THRESHOLD` | Low raw detection floor; session confidence is class-specific |
| `YOLO_DEVICE` | `auto`, `cpu`, `cuda`, `cuda:0`, or `mps` |
| `ENABLE_SUPPLEMENTAL_MODEL` | Enable extra COCO detector for common-object coverage |
| `ENABLE_TILED_DETECTION` | Enable periodic overlapping image tiles to improve small/distant object recall |
| `ENABLE_EYEWEAR_CROP_DETECTION` | Enable high-resolution person-head crops specifically for glasses/sunglasses |
| `ENABLE_BODY_POSE` | Enable YOLO pose/keypoint body-part observations |
| `ENABLE_MEDIAPIPE_HANDS` | Enable optional hand landmarks when MediaPipe is installed |
| `TARGET_CLASSES` | Restrict detection/storage to a subset (or `None` for all supported model classes) |
| `SESSION_GRACE_SECONDS` | Time-based disappearance grace before closing sessions |
| `SCENE_CLASSIFY_EVERY_N_FRAMES` | How often to run the (heavier) scene classifier |
| `MONGO_URI`, `MONGO_DB_NAME` | Where to store data |

---

## 5. Using a Different Dataset / Model

The system uses Open Images V7, optional COCO supplemental detection, optional body pose and MediaPipe hand landmarks, plus Places365 scene classification. The code is written so models can be swapped:

- **Detection dataset/model**: any Ultralytics-compatible YOLO checkpoint works — including a
  custom-trained model on **Open Images**, a custom dataset, or a fine-tuned COCO model. Just point
  `YOLO_MODEL_PATH` in `config.py` at your `.pt` file (place it in `models/object_detection_model/`)
  and update `TARGET_CLASSES` if the class names differ.
- **Scene dataset/model**: `scene_classifier.py` is isolated behind a `predict(frame) -> (label, confidence)`
  interface, so Places365 could be swapped for another scene-classification backbone (e.g. a
  custom-trained classifier on your own indoor/outdoor image set) without touching `main.py`.

---

## 6. How Sessions Work (avoiding duplicate records)

A new MongoDB record is **not** created every frame. Instead:

- **Detections**: each tracked object has an active session in memory. Sessions close after a
  time-based disappearance grace period plus a small missed-frame safety count, so closure behavior
  is less dependent on camera FPS.
- **Scenes**: the same idea — a scene session stays open as long as the same scene keeps winning
  classification; it closes (and a new one opens) when the scene changes.
- **Relationships**: a session stays open as long as the *same combination* of object classes is
  visible; it closes when that exact combination is no longer fully present.

This keeps the database compact (one row per real-world event) instead of one row per frame
(which at 30 FPS would be enormous and mostly redundant).

---

## 7. Error Handling

- **Camera not found / in use** → `main.py` logs a clear error and exits instead of crashing.
- **MongoDB unreachable** → camera processing continues. Session/scene/relationship writes are buffered
  in memory and retried with exponential backoff. Buffered records are visible in the dashboard and can
  still be included in the current-run PDF. The queue is not persistent across an application crash.
- **Places365 weights missing** → scene classification is disabled, everything else keeps working.
- **Stopping the camera** (`q`/`Esc`/`Ctrl+C`) always flushes any still-open sessions to MongoDB
  before exiting, so you don't lose the last in-progress detection.

---

## 8. Performance Notes

- The nano models target speed, but running Open Images, supplemental COCO, and pose can still be slow on CPU.
  Benchmark your computer. Disable optional models or lower image size/resolution if FPS is low; GPU is faster.
- Scene classification runs only every `SCENE_CLASSIFY_EVERY_N_FRAMES` frames (default 15) since
  it's the most expensive step per call — this keeps overall FPS high without needing GPU.
- To evaluate the system formally (as called out in the original spec): log YOLO's own `results.boxes`
  confidences for precision/recall/mAP/IoU studies against a labeled validation set, and log ByteTrack
  ID switches by comparing ground-truth tracks — this typically means running `main.py` against
  recorded video with a corresponding annotation file rather than a live webcam.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera index 0` | Try `CAMERA_INDEX = 1` in `config.py`, or close other apps using the webcam |
| Dashboard shows "Not connected to MongoDB" | Confirm `mongod` is running locally, or that `MONGO_URI` is set correctly |
| No scene shown in the overlay | Run `python download_places365.py` first |
| Low FPS | Disable optional models, lower `YOLO_IMAGE_SIZE`, lower camera resolution, or use a GPU |
| `ultralytics` download fails on first run | Check your internet connection; YOLO weights download from GitHub on first use |

## Tracking stability & validation (settings in `config.py`)

- `bytetrack_custom.yaml` – ByteTrack settings (low-score rescue pass, 60-frame lost buffer).
- `YOLO_CONFIDENCE_THRESHOLD` is the raw detector floor (0.08); what counts as a saved session is decided by class-specific thresholds and temporal confirmation.
- Every saved detection session carries `run_id` (one camera run) and `session_uid`; the PDF lists only the current run.
- `python diagnose.py --seconds 60 [--show]` prints raw ids vs confirmed sessions and rejection reasons without touching MongoDB. `python -m unittest discover -s tests` runs the logic tests.


## Expanded detection features (updated)

- **Primary detector:** bundled `yolov8n-oiv7.pt` Open Images V7 model. `TARGET_CLASSES = None` means all classes provided by that model are allowed.
- **Supplemental detector:** optional COCO nano model, auto-downloaded by Ultralytics when enabled, adds common-object coverage. Disable with `ENABLE_SUPPLEMENTAL_MODEL=0` in `.env` if CPU performance is too slow.
- **Raw detection debugging:** press **D** in the camera preview to toggle raw YOLO boxes. Orange boxes show raw detections; green boxes show objects that passed tracking/session confirmation. The diagnostic script reports raw detections, confidence, tracker IDs, and saved sessions separately.
- **Body pose:** enabled by default. Ultralytics downloads `yolov8n-pose.pt` on first startup and uses keypoints internally to draw smoothed boxes for torso, arms and legs only (no skeleton lines, eyes, or separate head box). Disable with `ENABLE_BODY_POSE=0` if performance is low.
- **Hands:** MediaPipe hand detection is a separate pipeline (not wrist-as-hand inference). It runs a full-frame pass plus upscaled crops around pose wrist hints to improve detection of small/distant hands. Detection/tracking confidence is set lower (0.25). The preview shows hand landmark dots and a hand box, with no skeleton lines. The status reads ACTIVE/count, SEARCHING, or UNAVAILABLE; errors are displayed when initialization or inference fails. Install with `python -m pip install -r requirements-hands.txt` (recommended Python 3.11/3.12).
- **Small-object refinement:** every six frames, the primary Open Images model scans overlapping image tiles, allowing smaller/distant objects to occupy more pixels. Crop detections are deduplicated and assigned stable lightweight track IDs. Toggle with `ENABLE_TILED_DETECTION=0` if CPU FPS is too low.
- **Glasses/sunglasses refinement:** every four frames, up to two detected person boxes generate enlarged upper-head crops. The primary Open Images model runs an eyewear-only pass on those crops, using a lower crop confidence threshold and a context-aware confirmation rule. The COCO supplemental model does not contain glasses classes, so this refinement uses the Open Images model. Toggle with `ENABLE_EYEWEAR_CROP_DETECTION=0` to disable it.
- **Small-object/body-part confirmation:** class-aware lower thresholds are used for phones, body parts and small daily-use objects. They still require repeated observations to avoid saving one-frame false positives.
- **Dashboard:** Live tab refreshes automatically every five seconds and displays device, raw detections, pose and hand feature status.
- **Database outage behavior:** MongoDB retries use exponential backoff rather than retrying on every frame.
- **Paths:** model, tracker, scene-model and PDF output paths are project-root based. Run scripts from any working directory.
- **Privacy:** `.env`, Git metadata and local PDF reports are not included in the distribution ZIP. Copy `.env.example` to `.env` and set your own credentials.

### Setup on Windows

1. Install Python 3.11 or 3.12 (recommended for the optional MediaPipe dependency).
2. Run `setup.bat` to create `.venv` and install required packages.
3. Copy `.env.example` to `.env` and configure MongoDB/admin password.
4. Run `start_detection.bat`.
5. Press **D** to toggle raw detections; **Q** or **Esc** exits.

### Performance note

Periodic tile and head-crop refinement plus the supplemental detector and body pose can reduce FPS on CPU. For a slower computer, set `ENABLE_SUPPLEMENTAL_MODEL=0` and/or `ENABLE_BODY_POSE=0`, reduce `YOLO_IMAGE_SIZE` to 640, or use a CUDA GPU. The app prioritizes coverage when these options are enabled; benchmark on your camera before keeping all modules enabled.
