# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run serve_model.py when the container launches
CMD ["python", "serve_model.py"]
