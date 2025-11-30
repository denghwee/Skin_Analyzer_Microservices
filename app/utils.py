import base64
from io import BytesIO
from PIL import Image, ImageDraw
import cloudinary.uploader
import tempfile
def crop_regions(image, detections):
    crops = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
    return crops

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1 - 10), f"{det['class']} ({det['confidence']:.2f})", fill="red")
    return image

def image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return encoded
def upload_base64_to_cloudinary(base64_str, folder="skin_analysis"):
    """
    Nhận base64 → upload lên Cloudinary → trả về URL.
    """

    # Loại bỏ prefix nếu có dạng data:image/png;base64,...
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]

    image_bytes = base64.b64decode(base64_str)

    # Dùng file tạm để upload
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(image_bytes)
        temp.flush()

        result = cloudinary.uploader.upload(
            temp.name,
            folder=folder,
            resource_type="image"
        )

        return result["secure_url"]