FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eval_from_hf.py .
COPY utils.py .

# Automatically run evaluation on container start
CMD ["python", "eval_from_hf.py"]
