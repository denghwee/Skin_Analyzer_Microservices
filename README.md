## Skin Analyzer Web App

An interactive Flask web application that helps screen facial or skin images using a two-stage AI workflow. The service detects regions of interest with a YOLO model, crops them, and runs an EfficientNet classifier to suggest likely dermatological conditions. The interface provides a drag-and-drop uploader, instant preview, and a summarized report of detections.

## Features
- Upload or drag-and-drop a skin image and preview it before analysis.
- YOLO-based object detection to locate relevant skin regions.
- EfficientNet-based disease classification for each detected region.
- Annotated output image plus a clean summary table with confidence scores.
- Lightweight `/health` endpoint for service monitoring.

## Tech Stack
- Python 3.11
- Flask
- Pillow
- PyTorch
- Ultralytics YOLO

## Project Structure
```
.
├── app
│   ├── __init__.py          # Flask factory and configuration loading
│   ├── routes.py            # HTTP routes and request handling
│   ├── classification_service.py
│   ├── objectdetection_service.py
│   ├── utils.py             # Helper functions (cropping, drawing boxes)
│   ├── templates            # HTML templates (index, result)
│   └── static               # CSS, JS, generated result image
├── requirements.txt
└── run.py                   # App entry point
```

Pretrained model weights are stored in `app/models/` and are loaded at runtime; do not commit replacements without confirming licensing.

## Getting Started
1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate # macOS / Linux
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the development server**
   ```bash
   python run.py
   ```
   By default, the app listens on `http://0.0.0.0:5000`. Change `MICRO_HOST`, `MICRO_PORT`, or `MICRO_DEBUG` environment variables as needed.

## Usage
1. Open the homepage.
2. Upload or drop a face/skin photo (PNG or JPG, ≤10 MB).
3. Click `Analyze Now` and wait for processing.
4. Review the annotated image and detection table on the results page.

## API Endpoints
- `GET /` – Upload interface.
- `POST /analyze` – Accepts multipart image uploads; runs detection and classification.
- `GET /health` – Returns `{ "status": "ok" }` when the service is healthy.

## Notes
- Ensure GPU support is configured if running heavy workloads; otherwise the models will run on CPU.
- The shipped model weights are for demo purposes. Validate accuracy and compliance before clinical use.
- Delete `app/static/result.jpg` between runs if you need to clear prior results.


