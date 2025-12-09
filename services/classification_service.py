import os
from typing import Any, Dict

from flask import Flask, request
from PIL import Image

from services.classification.inference import classify_image


def _load_image() -> Image.Image:
    file_storage = request.files.get("image")
    if not file_storage:
        raise ValueError("Missing file field 'image'")
    return Image.open(file_storage.stream).convert("RGB")


def create_app() -> Flask:
    service = Flask(__name__)

    @service.route("/health", methods=["GET"])
    def health() -> Dict[str, str]:
        return {"status": "ok", "service": "classification"}

    @service.route("/classify", methods=["POST"])
    def classify() -> Dict[str, Any]:
        try:
            image = _load_image()
        except ValueError as error:
            return {"error": str(error)}, 400

        prediction = classify_image(image)
        return {"prediction": prediction}

    return service


app = create_app()


def main() -> None:
    host = os.getenv("CLASSIFICATION_HOST", "0.0.0.0")
    port = int(os.getenv("CLASSIFICATION_PORT", "5002"))
    debug = os.getenv("CLASSIFICATION_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

