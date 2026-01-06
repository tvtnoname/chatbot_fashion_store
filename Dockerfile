FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Do not copy code yet to avoid rebuilding on every code change during dev, 
# relying on volume mount instead.
# COPY . .

CMD ["python", "main.py"]
