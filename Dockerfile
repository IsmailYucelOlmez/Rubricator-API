FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1

# Render's proxy puts the real client IP first in X-Forwarded-For; without these
# flags every request looks like it came from the proxy and the per-IP rate
# limits turn into one bucket shared by all users.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
