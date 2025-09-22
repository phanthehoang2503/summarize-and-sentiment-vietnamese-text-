# Hệ Thống Phân Tích Văn Bản Tiếng Việt

Hệ thống tóm tắt văn bản và phân tích cảm xúc tiếng Việt sử dụng mô hình Transformer.

## Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- RAM 8GB+ (khuyến nghị 16GB để huấn luyện)
- GPU hỗ trợ CUDA (tùy chọn, để xử lý nhanh hơn)
- Dung lượng trống 10GB+

## Cài Đặt

1. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

## Cấu Trúc Thư Mục

```
MajorProject/
├── app/                    # Ứng dụng web
│   ├── main.py            # Điểm vào ứng dụng Flask
│   ├── core/              # Cấu hình cốt lõi
│   └── api/               # Các endpoint API
├── data/
│   ├── raw/               # Dữ liệu gốc
│   ├── processed/         # Dữ liệu đã làm sạch
│   └── vietnamese-stopwords.txt
├── models/                # Checkpoint mô hình đã huấn luyện
│   ├── sentiment/         # Mô hình phân tích cảm xúc
│   └── summarizer/        # Mô hình tóm tắt
├── scripts/               # Script xử lý và huấn luyện
│   ├── preprocessing.py   # Tiền xử lý dữ liệu
│   ├── train_summarizer.py
│   ├── evaluate_model.py
│   └── demos/             # Script demo
├── notebooks/             # Jupyter notebooks
├── src/                   # Module mã nguồn
├── static/                # Tài nguyên web
├── templates/             # Template HTML
└── logs/                  # Log ứng dụng
```

## Tiền Xử Lý Dữ Liệu

Xử lý dữ liệu thô thành tập dữ liệu sạch:

```bash
python scripts/preprocessing.py
```

Quá trình này sẽ:
- Làm sạch và lấy mẫu dữ liệu tóm tắt (mặc định 50%)
- Làm sạch dữ liệu cảm xúc với việc loại bỏ stopword
- Lưu file đã xử lý vào `data/processed/`

## Huấn Luyện Mô Hình

Huấn luyện mô hình tóm tắt:
```bash
python scripts/train_summarizer.py
```

Huấn luyện mô hình phân tích cảm xúc:
```bash
python scripts/train_sentiment.py
```

## Chạy Ứng Dụng Web

Khởi động server Flask:
```bash
python app/main.py
```

Hoặc sử dụng phương pháp thay thế:
```bash
python -m flask --app app.main run
```

Ứng dụng web sẽ có sẵn tại: http://localhost:5000

## Sử Dụng Dòng Lệnh

### Script Demo

Kiểm tra phân tích cảm xúc:
```bash
python scripts/demos/demo_sentiment.py
```

Kiểm tra tóm tắt văn bản:
```bash
python scripts/demos/demo_summarizer.py
```

Kiểm tra quy trình đầy đủ:
```bash
python scripts/demos/demo_pipeline.py
```

### Jupyter Notebooks

Khởi động Jupyter và mở notebooks:
```bash
jupyter notebook notebooks/
```

Các notebook có sẵn:
- `00_data_analysis.ipynb` - Khám phá dữ liệu
- `01_processing.ipynb` - Tiền xử lý dữ liệu
- `02_train_summarizer.ipynb` - Huấn luyện tóm tắt
- `03_train_sentiment.ipynb` - Huấn luyện cảm xúc

## Cấu Hình Mô Hình

Mô hình được cấu hình trong `params.yaml`:

- **Mô hình Cảm xúc**: PhoBERT-base (vinai/phobert-base)
- **Mô hình Tóm tắt**: ViT5-base (VietAI/vit5-base)
- **Thư mục Cache**: `cache/`
- **Checkpoint Mô hình**: `models/`

## Yêu Cầu Dữ Liệu

Đặt tập dữ liệu vào `data/raw/`:
- `data_sentiment.csv` - Dữ liệu phân tích cảm xúc (cột: comment, label)
- `data_summary.csv` - Dữ liệu tóm tắt (cột: Text, Summary)

## Đánh Giá

Đánh giá mô hình đã huấn luyện:
```bash
python scripts/evaluate_model.py
```

## Dọn Dẹp

Xóa file tạm và cache:
```bash
python scripts/cleanup.py
```

## Khắc Phục Sự Cố

**ImportError**: Đảm bảo tất cả thư viện đã được cài đặt:
```bash
pip install -r requirements.txt
```

## API Endpoints

Khi chạy ứng dụng web:

- `GET /` - Giao diện chính
- `POST /analyze` - Endpoint phân tích văn bản
- `POST /summarize` - Tóm tắt văn bản
- `POST /sentiment` - Phân tích cảm xúc
