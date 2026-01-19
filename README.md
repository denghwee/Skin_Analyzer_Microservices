# Skin Analyzer Microservice

Microservice Flask để phân tích da sử dụng AI, phát hiện các vấn đề về da và đưa ra gợi ý chăm sóc sức khỏe. Service sử dụng YOLO v10 để phát hiện vùng da và EfficientNet để phân loại các tình trạng da.

## Tính năng

- **Phát hiện đối tượng**: Sử dụng YOLO v10 để phát hiện các vùng da có vấn đề
- **Phân loại bệnh lý**: Sử dụng EfficientNet để phân loại các tình trạng da cụ thể
- **Lưu trữ kết quả**: Lưu lịch sử phân tích vào database
- **Gợi ý sức khỏe**: Tự động tạo gợi ý về lối sống và chế độ ăn uống
- **Ghi chú bác sĩ**: Cho phép bác sĩ thêm ghi chú vào kết quả phân tích
- **Xác thực JWT**: Bảo mật API bằng JWT authentication
- **Tích hợp Cloudinary**: Lưu trữ và quản lý hình ảnh trên Cloudinary

## Tech Stack

- **Python 3.11**
- **Flask** - Web framework
- **Flask-SQLAlchemy** - ORM cho database
- **Flask-Migrate** - Database migrations
- **Flask-JWT-Extended** - JWT authentication
- **PyTorch** - Deep learning framework
- **ONNX Runtime** - Inference engine cho AI models
- **Ultralytics YOLO** - Object detection
- **Cloudinary** - Image hosting và CDN
- **Alembic** - Database migration tool

## Cấu trúc dự án

```
.
├── app/
│   ├── __init__.py              # Flask factory và cấu hình
│   ├── config.py                # Cấu hình ứng dụng
│   ├── controllers/             # API controllers
│   │   └── analysis_controller.py
│   ├── models/                  # Database models và DTOs
│   │   ├── analysis_entity.py   # SQLAlchemy entity
│   │   ├── analysis_model.py    # Response model
│   │   └── analyze_request_dto.py
│   ├── models_AI/               # AI model files
│   │   ├── efficientnet_b2_skin.onnx
│   │   ├── efficientnet_b2_skin.pth
│   │   ├── yolov10_skin.onnx
│   │   └── yolov10_skin.pt
│   ├── routes/                  # Route handlers
│   │   └── routes.py
│   ├── services/                # Business logic
│   │   └── analysis_service.py
│   ├── services_AI/             # AI services
│   │   ├── classification/      # Classification service
│   │   └── objectdetection/     # Object detection service
│   └── utils/                   # Utility functions
│       ├── utils.py
│       └── health_info.py
├── migrations/                  # Database migrations
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── run.py                       # Application entry point
```

## Cài đặt

### 1. Tạo và kích hoạt virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình môi trường

Tạo file `.env` với các biến môi trường sau:

```env
# Database
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/skin_analyzer

# JWT
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret-key

# Cloudinary (optional)
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
# Hoặc sử dụng CLOUDINARY_URL=cloudinary://api_key:api_secret@cloud_name

# Server
MICRO_HOST=0.0.0.0
MICRO_PORT=5001
MICRO_DEBUG=0
```

### 4. Chạy database migrations

```bash
flask db upgrade
```

### 5. Chạy development server

```bash
python run.py
```

Service sẽ chạy tại `http://0.0.0.0:5001` (hoặc port được cấu hình trong biến môi trường).

## API Endpoints

### Health Check
- **GET** `/api/v1/health` - Kiểm tra trạng thái service
  - Response: `{"status": "ok"}`

### Analysis
- **POST** `/api/v1/analyze` - Phân tích hình ảnh da
  - **Authentication**: JWT required
  - **Request**: Multipart form data với field `image`
  - **Response**: Kết quả phân tích với annotated image URL, detections, health info và lifestyle suggestions

### Analysis Management
- **POST** `/api/v1/analysis/predict` - Lưu kết quả phân tích đơn giản
  - **Authentication**: JWT required
  - **Body**: 
    ```json
    {
      "annotatedImageUrl": "string",
      "aiDiagnosis": "string",
      "aiConfidence": 0.95,
      "suggestions": {
        "lifestyle": ["..."],
        "diet": ["..."]
      }
    }
    ```

- **POST** `/api/v1/analysis/save-ai-result` - Lưu kết quả đầy đủ từ AI service
  - **Authentication**: JWT required
  - **Body**: Full analysis result từ `/api/v1/analyze`

- **GET** `/api/v1/analysis/history` - Lấy lịch sử phân tích của user
  - **Authentication**: JWT required
  - **Response**: Array of analysis records

- **PATCH** `/api/v1/analysis/<record_id>/doctor-note` - Cập nhật ghi chú bác sĩ
  - **Authentication**: JWT required
  - **Body**: 
    ```json
    {
      "doctorNote": "string"
    }
    ```

## Database Schema

### health_analysis
- `id` (Integer, Primary Key)
- `user_id` (Integer, Not Null)
- `analysis_image_url` (String(500), Not Null)
- `ai_diagnosis` (String(255), Not Null)
- `ai_confidence` (Float, Not Null)
- `suggestions` (JSON, Not Null)
- `doctor_note` (Text, Nullable)
- `created_at` (DateTime)
- `doctor_updated_at` (DateTime, Nullable)

## AI Models

Service sử dụng 2 loại AI models:

1. **YOLO v10** (`yolov10_skin.pt` / `yolov10_skin.onnx`)
   - Phát hiện các vùng da có vấn đề
   - Classes: Dark Circle, Eyebag, Acne Scar, Blackhead, Dark spot, Freckle, Melasma, Nodules, Papules, Pustules, Skinredness, Vascular, Whitehead, Wrinkle, etc.

2. **EfficientNet-B2** (`efficientnet_b2_skin.pth` / `efficientnet_b2_skin.onnx`)
   - Phân loại các tình trạng da cụ thể
   - Chỉ chạy cho các classes: acne scar, melasma, nodules, papules, pustules, skinredness, vascular

## Docker

Build và chạy với Docker:

```bash
docker build -t skin-analyzer .
docker run -p 5001:5001 --env-file .env skin-analyzer
```

## Cloudinary (Optional)

Service hỗ trợ upload hình ảnh lên Cloudinary. Nếu không cấu hình Cloudinary, service sẽ trả về base64 data URI cho hình ảnh đã annotate.

Để bật Cloudinary, cấu hình một trong các cách sau:
- Set biến môi trường `CLOUDINARY_URL`
- Hoặc set các biến riêng lẻ: `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`

## Lưu ý

- Service yêu cầu JWT token trong header `Authorization: Bearer <token>` cho hầu hết các endpoints
- JWT token phải chứa `userId` trong claims
- Model weights được lưu trong `app/models_AI/` - không commit thay thế mà không xác nhận license
- Đảm bảo GPU được cấu hình nếu chạy workloads nặng; nếu không models sẽ chạy trên CPU
- Model weights hiện tại chỉ dùng cho mục đích demo. Cần validate accuracy và compliance trước khi sử dụng trong lâm sàng

## Development

### Chạy migrations mới

```bash
flask db migrate -m "migration message"
flask db upgrade
```

### Cấu hình debug mode

Set `MICRO_DEBUG=1` trong `.env` để bật debug mode.
