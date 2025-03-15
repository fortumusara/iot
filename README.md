Here’s how to run your entire IoT Fire Alarm Backend Data Engineering Project using Docker Compose on your Windows machine 🚀.
________________________________________
1️⃣ Install Prerequisites
Ensure the following are installed:
✅ Docker Desktop → Download
✅ Git → Download
✅ Python → Download
✅ Node.js (for Quasar) → Download
✅ Bitbucket CLI (Optional) → Download
________________________________________
2️⃣ Clone Your Project Repo
bash
CopyEdit
git clone https://bitbucket.org/your-repo.git
cd your-repo
________________________________________
3️⃣ Create a docker-compose.yml File
Create a new file docker-compose.yml inside your project root:
yaml
CopyEdit
version: "3.8"

services:
  mysql:
    image: mysql:latest
    container_name: mysql_db
    restart: always
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: fire_alarm_db
    volumes:
      - mysql_data:/var/lib/mysql

  minio:
    image: quay.io/minio/minio
    container_name: minio_s3
    restart: always
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  backend:
    build: ./backend
    container_name: django_backend
    restart: always
    depends_on:
      - mysql
    ports:
      - "8000:8000"
    environment:
      DATABASE_HOST: mysql
      DATABASE_USER: root
      DATABASE_PASSWORD: root
      DATABASE_NAME: fire_alarm_db
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    container_name: quasar_frontend
    restart: always
    depends_on:
      - backend
    ports:
      - "9002:9000"
    volumes:
      - ./frontend:/app
    command: ["quasar", "dev"]

  glue-etl:
    build: ./etl
    container_name: glue_etl
    restart: always
    depends_on:
      - backend
    volumes:
      - ./etl:/app
    command: ["spark-submit", "etl_script.py"]

volumes:
  mysql_data:
  minio_data:
________________________________________
4️⃣ Add a Dockerfile for Each Service
🔹 Backend (backend/Dockerfile)
dockerfile
CopyEdit
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
________________________________________
🔹 Frontend (frontend/Dockerfile)
dockerfile
CopyEdit
FROM node:18
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
CMD ["quasar", "dev"]
________________________________________
🔹 ETL (etl/Dockerfile)
dockerfile
CopyEdit
FROM apache/spark:latest
WORKDIR /app
COPY . .
CMD ["spark-submit", "etl_script.py"]
________________________________________
5️⃣ Run the Entire Project
Run the following command inside your project directory:
bash
CopyEdit
docker-compose up --build
________________________________________
6️⃣ Access Your Services
🚀 Django API (Backend): http://localhost:8000
🌐 Quasar Frontend: http://localhost:9002
💾 MySQL Database: mysql://root:root@localhost:3306/fire_alarm_db
📦 MinIO (S3 Alternative): http://localhost:9001
🔥 Glue ETL (Simulated with Spark): Runs automatically in the container
________________________________________
🎯 Now Everything Runs with One Command!
Would you like Airflow DAGs to be added to this setup? 

