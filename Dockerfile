# Use a lightweight official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to optimize Python inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for text parsing and network utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements document first to leverage Docker's caching layers
COPY requirements.txt .

# Install Python packages inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your local application code into the container
COPY . .

# Expose the network ports used by FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Copy a startup shell script to run both applications simultaneously
CMD ["sh", "-c", "python main.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]