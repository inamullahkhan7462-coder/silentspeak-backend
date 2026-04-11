# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install
# Note: We use tensorflow-cpu to keep the image small and fast
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port Google Cloud Run uses
EXPOSE 8080

# Command to run the server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]