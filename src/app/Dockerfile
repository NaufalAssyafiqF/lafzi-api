FROM python:3.11-slim 

ENV PYTHONUNBUFFERED True
ENV APP_HOME /src/app

WORKDIR $APP_HOME

# Copy requirements.txt terlebih dahulu untuk memanfaatkan cache layer
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi
COPY . .

# Jalankan Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
