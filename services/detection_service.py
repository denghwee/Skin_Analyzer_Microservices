import os
from typing import Any, Dict, List

from flask import Flask, request
from PIL import Image

from services.detection.inference import detect_objects


def _load_image() -> Image.Image:
    file_storage = request.files.get("image")
    if not file_storage:
        raise ValueError("Missing file field 'image'")
    return Image.open(file_storage.stream).convert("RGB")


def create_app() -> Flask:
    service = Flask(__name__)

    @service.route("/health", methods=["GET"])
    def health() -> Dict[str, str]:
        return {"status": "ok", "service": "detection"}

    @service.route("/detect", methods=["POST"])
    def detect() -> Dict[str, Any]:
        try:
            image = _load_image()
        except ValueError as error:
            return {"error": str(error)}, 400

        detections: List[Dict[str, Any]] = detect_objects(image)
        return {"detections": detections}

    return service


app = create_app()


def main() -> None:
    host = os.getenv("DETECTION_HOST", "0.0.0.0")
    port = int(os.getenv("DETECTION_PORT", "5001"))
    debug = os.getenv("DETECTION_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

