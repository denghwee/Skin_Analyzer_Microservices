## Skin Analyzer Platform

This repository delivers a cloud-friendly skin analysis platform with a two-stage inference pipeline (YOLO detection → EfficientNet classification). The codebase is organised as dedicated microservices so you can scale the heavy models independently when running on the cloud.

### Key Capabilities
- Drag-and-drop image uploader with annotated output and detection summary
- YOLO-based localization of dermatological regions of interest
- EfficientNet classification only for detections that need it (configurable)
- Health checks per service and graceful error propagation
- Configurable service endpoints and timeouts via environment variables

### Layout Overview
```
.
├── app/                       # Gateway web application
│   ├── __init__.py            # Flask factory
│   ├── routes.py              # User-facing routes & orchestration
│   ├── service_clients.py     # Detection/classification service adapters
│   ├── templates/             # Jinja templates
│   ├── static/                # Front-end assets
│   └── models/                # Shared assets (e.g., class metadata)
├── services/
│   ├── detection/             # YOLO inference package
│   │   └── inference.py
│   ├── classification/        # EfficientNet inference package
│   │   └── inference.py
│   ├── food_detection/        # YOLO inference for food items (new)
│   │   └── inference.py
│   ├── detection_service.py   # Detection microservice
│   └── classification_service.py
├── services/food_detection_service.py  # Food detection microservice (new)
├── run.py                     # Local dev entry point
├── wsgi.py                    # WSGI entry point for Gunicorn
├── Dockerfile                 # Container recipe (configurable service target)
└── requirements.txt
```

### Configuration
Each service is driven entirely by environment variables:

**Gateway (`app/`):**

| Variable | Purpose | Default |
|----------|---------|---------|
| `DETECTION_SERVICE_URL` | Detection endpoint | `http://localhost:5001/detect` |
| `CLASSIFICATION_SERVICE_URL` | Classification endpoint | `http://localhost:5002/classify` |
| `SERVICE_REQUEST_TIMEOUT` | HTTP timeout (seconds) | `30` |
| `CLASSES_REQUIRING_CLASSIFICATION` | CSV of detection labels to forward to classifier | `acne scar,melasma,...` |
| `FOOD_DETECTION_SERVICE_URL` | Food detection endpoint (gateway calls this) | `http://localhost:5003/detect` |
| `FOOD_CONFIDENCE_THRESHOLD` | Confidence threshold for food pipeline outputs | `0.5` |

**Detection service:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `YOLO_MODEL_PATH` | Weight file | `app/models/yolov11_skin.pt` |
| `YOLO_CONFIDENCE` | Detection confidence threshold | `0.25` |
| `YOLO_IMAGE_SIZE` | Inference image size | `640` |
| `YOLO_DEVICE` | Device string for Ultralytics | `cpu` |

**Food detection service (new):**

| Variable | Purpose | Default |
|----------|---------|---------|
| `FOOD_YOLO_MODEL_PATH` | Food YOLO weight file | `app/models/food_detection.pt` |
| `FOOD_YOLO_CONFIDENCE` | Detection confidence threshold for food model | `0.25` |
| `FOOD_YOLO_IMAGE_SIZE` | Inference image size for food model | `640` |
| `FOOD_YOLO_DEVICE` | Device string for Ultralytics (food) | `cpu` |

**Classification service:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `EFFICIENTNET_PATH` | Weight file | `app/models/efficientnet_b2_skin.pth` |
| `CLASSIFIER_DEVICE` | Torch device | auto (`cuda` if available else `cpu`) |
| `CLASSIFIER_IMG_HEIGHT`, `CLASSIFIER_IMG_WIDTH` | Resize dimensions | `224` |
| `CLASS_NAMES` | CSV class list (used for labels) | `none,acne,...` |
| `NUM_CLASSES` | Override output layer size | derived from `CLASS_NAMES` |

See the source modules for additional tuning knobs.

### Local Development
```bash
python -m venv venv
venv\Scripts\activate           # or source venv/bin/activate
pip install -r requirements.txt
python -m services.detection_service
python -m services.classification_service
python -m services.food_detection_service    # start the food detection service (new)
python run.py                   # serves the gateway on :5000
```

Start the detection and classification services first so the gateway can reach their endpoints.
If you use the food pipeline, start the food detection service as well.

### Container Deployment
The `Dockerfile` is parameterized via environment variables:

```bash
docker build -t skin-gateway .
docker run -e APP_MODULE=wsgi:app -e PORT=5000 skin-gateway         # Gateway
docker run -e APP_MODULE=services.detection_service:app -e PORT=5001 skin-gateway
docker run -e APP_MODULE=services.classification_service:app -e PORT=5002 skin-gateway
```

Adjust `WORKERS` or `GUNICORN_TIMEOUT` to fit your workload. On GPU-enabled hosts, mount the appropriate drivers and set `INFERENCE_DEVICE=cuda`.

> When deploying inside Docker/Kubernetes, override `DETECTION_SERVICE_URL` and `CLASSIFICATION_SERVICE_URL` to use the in-cluster service names (e.g., `http://detection:5001/detect`).

### Notes
- Ensure model weights are available at runtime (mount volumes or bake into the image).
- Food model: provide `app/models/food_detection.pt` or set `FOOD_YOLO_MODEL_PATH` to point to your weights.
- Apply TLS and authentication at the gateway or via your cloud ingress when exposing the services publicly.
- Add observability (e.g., Prometheus, log aggregation) when running microservices at scale.

For more detail on the orchestration logic, read `app/routes.py` and `app/service_clients.py`.

### Food detection pipeline (gateway)

The gateway now exposes a dedicated endpoint for food detection:

- `POST /detect_food` — accepts multipart form with `image` and returns:
	- `top_food`: single detection with highest confidence (if any)
	- `foods_above_threshold`: list of detected food items with `confidence >= FOOD_CONFIDENCE_THRESHOLD`
	- `all_foods`: all detections returned by the food service
	- `annotated_image_base64`: image with boxes (when bbox present)

Sample request (curl / PowerShell):

```powershell
curl -X POST -F "image=@path/to/sample.jpg" http://localhost:5000/detect_food
```

Or PowerShell `Invoke-RestMethod`:

```powershell
$img = "path\to\sample.jpg"
Invoke-RestMethod -Uri http://localhost:5000/detect_food -Method Post -Form @{ image = Get-Item $img } | ConvertTo-Json
```

The gateway calls the configured `FOOD_DETECTION_SERVICE_URL` (default `http://localhost:5003/detect`).
