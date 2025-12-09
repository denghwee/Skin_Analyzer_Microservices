import io
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from .config import Config


class ServiceCallError(RuntimeError):
    """Raised when a downstream service call fails."""


def _serialize_image(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def call_detection_service(image: Image.Image) -> List[Dict[str, Any]]:
    files = {"image": ("image.png", _serialize_image(image), "image/png")}
    try:
        response = requests.post(
            Config.DETECTION_SERVICE_URL,
            files=files,
            timeout=Config.SERVICE_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ServiceCallError(f"Detection service call failed: {exc}") from exc

    payload = response.json()
    return payload.get("detections", [])


def call_classification_service(image: Image.Image) -> Optional[Dict[str, Any]]:
    files = {"image": ("region.png", _serialize_image(image), "image/png")}
    try:
        response = requests.post(
            Config.CLASSIFICATION_SERVICE_URL,
            files=files,
            timeout=Config.SERVICE_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ServiceCallError(f"Classification service call failed: {exc}") from exc

    payload = response.json()
    return payload.get("prediction")


def call_food_detection_service(image: Image.Image) -> List[Dict[str, Any]]:
    """Call the food detection service and return list of detected foods.

    Returns a list of dicts with keys like 'class', 'confidence', 'bbox' (if provided).
    """
    files = {"image": ("image.png", _serialize_image(image), "image/png")}
    try:
        response = requests.post(
            Config.FOOD_DETECTION_SERVICE_URL,
            files=files,
            timeout=Config.SERVICE_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ServiceCallError(f"Food detection service call failed: {exc}") from exc

    payload = response.json()
    # Support both 'foods' and 'detections' keys for backward compatibility
    return payload.get("foods") or payload.get("detections") or []

