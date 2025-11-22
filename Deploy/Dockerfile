# Use official Python image (lightweight)
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system libraries (for compiling e.g. numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy only requirements (so Docker "caches" library installation)
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy entire project code
COPY . .

# 3. Install your package (Dragon) inside Docker
RUN pip install .

# Expose port 8000 (where FastAPI runs)
EXPOSE 8000

# Start server (Note: path to API/server.py)
# Use python -m so system paths work correctly
CMD ["python", "API/server.py"]