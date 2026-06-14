# ใช้ Python เวอร์ชัน 3.10
FROM python:3.10-slim

WORKDIR /app

# ก๊อปปี้ไฟล์ requirements และติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้ไฟล์ทั้งหมด (data, public, main.py) เข้าเซิร์ฟเวอร์
COPY . .

# เปิดพอร์ตรับคนเข้าเว็บ
EXPOSE 8000

# คำสั่งสตาร์ทหลังบ้าน
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]