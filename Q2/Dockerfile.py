FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY requirements.txt .

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies
RUN uv pip install --system --no-cache -r requirements.txt

CMD ["bash"]