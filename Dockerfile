FROM python:3.9-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make port 8000 available
EXPOSE 8000

# Set environment variables
ENV PORT=8000

# Command to run when the container starts
CMD gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app
