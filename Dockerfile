# Use official Python slim image
FROM python:3.13-slim
 
# Install Tesseract OCR and dependencies
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev && rm -rf /var/lib/apt/lists/*
 
# Set working directory in container
WORKDIR /app
 
# Copy all project files into container
COPY . .
 
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
 
# Expose port (Render default)
EXPOSE 10000
 
# Start Flask app using Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000"]