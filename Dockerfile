# 1. ใช้ Python เป็นเบสหลักในการรัน
FROM python:3.10-slim

# 2. ติดตั้ง Node.js และเครื่องมือสำหรับดาวน์โหลด dependencies
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://nodesource.com | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. คัดลอกและติดตั้ง Dependencies ของ Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. คัดลอกและติดตั้ง Dependencies ของ Node.js (ถ้ามีไฟล์ package.json)
COPY package*.json ./
RUN npm install

# 5. คัดลอกไฟล์ทั้งหมดในโปรเจ็กต์เข้ามา
COPY . .

# 6. เปิดพอร์ตที่ใช้งาน (เปิดเผื่อไว้ทั้งคู่หรือพอร์ตหลักที่คุยกับหลังบ้าน)
EXPOSE 8000
EXPOSE 3000

# 7. สั่งให้ทั้ง Uvicorn และ Node.js รันขึ้นมาพร้อมกัน
CMD node server.js ; uvicorn main:app --host 0.0.0.0 --port 8000