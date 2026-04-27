#!/bin/bash

echo "===================================="
echo "BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH"
echo "===================================="

# Bước 1: Tiền xử lý dữ liệu
echo "Đang tiền xử lý dữ liệu..."
python scripts/preprocess_data.py

# Bước 2: Huấn luyện mô hình
echo "Đang huấn luyện mô hình..."
python scripts/train.py

echo "===================================="
echo "HOÀN THÀNH HUẤN LUYỆN"
echo "===================================="