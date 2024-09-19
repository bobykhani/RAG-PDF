# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose the port (if you have a FastAPI/Flask service)
# EXPOSE 8000

# Command to run the application
# If you're using a specific script, update the script name accordingly
CMD ["python", "src/main.py"]
