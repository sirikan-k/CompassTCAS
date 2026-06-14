# ใช้ Python เป็นเบส
FROM python:3.10-slim

WORKDIR /app

# คัดลอกและติดตั้ง Dependencies ของ Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดในโปรเจ็กต์เข้ามา (รวมไฟล์ data และ main.py)
COPY . .

# สั่งให้ Uvicorn รัน
CMD uvicorn main:app --host 0.0.0.0 --port $PORT