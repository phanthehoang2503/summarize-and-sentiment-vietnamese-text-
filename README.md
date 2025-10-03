# Hệ Thống Phân Tích Văn Bản Tiếng Việt

Hệ thống tóm tắt văn bản và phân tích cảm xúc tiếng Việt sử dụng mô hình Transformer.

## Yêu Cầu Hệ Thống

- Python 3.8+ trở lên
- GPU hỗ trợ CUDA

## Cài Đặt

1. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

## Cấu Trúc Thư Mục

```
MajorProject/
├── app/                    # Ứng dụng web
│   ├── main.py            
│   ├── core/              
│   └── api/               
├── data/
│   ├── raw/               # Dữ liệu gốc
│   ├── processed/         # Dữ liệu đã được xử lý
│   └── vietnamese-stopwords.txt
├── models/                # Checkpoint các mô hình đã huấn luyện
│   ├── sentiment/         # Mô hình phân tích
│   └── summarizer/        # Mô hình tóm tắt
├── scripts/               # Script xử lý và huấn luyện
│   ├── preprocessing.py   
│   ├── train_summarizer.py
│   ├── evaluate_model.py
│   └── demos/             
├── notebooks/             
├── src/                   
├── static/                # Web
├── templates/             # HTML
└── logs/                  
```

## Tiền Xử Lý Dữ Liệu

Xử lý dữ liệu thô thành tập dữ liệu sạch:

```bash
python scripts/preprocessing.py
```

## Huấn Luyện Mô Hình

Huấn luyện mô hình tóm tắt:
```bash
python scripts/train_summarizer.py
```

Huấn luyện mô hình phân tích:
```bash
python scripts/train_sentiment.py
```

## Chạy Ứng Dụng Web

Khởi động server Flask:
```bash
python app/main.py
```

Truy cập: http://localhost:5000

## Demo CLI

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

## Yêu Cầu Dữ Liệu

Đặt tập dữ liệu vào `data/raw/`:
- `data_sentiment.csv` - Dữ liệu phân tích cảm xúc (cột: comment, label)
- `data_summary.csv` - Dữ liệu tóm tắt (cột: Text, Summary)


## Dọn Dẹp

Xóa file tạm và cache:
```bash
python scripts/cleanup.py
```

## API Endpoints

Khi chạy ứng dụng web:

- `GET /` - Giao diện chính
- `POST /analyze` - Endpoint phân tích văn bản
- `POST /summarize` - Tóm tắt văn bản
- `POST /sentiment` - Phân tích cảm xúc