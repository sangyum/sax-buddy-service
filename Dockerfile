# Use a lightweight Python 3.11 image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install uv and project dependencies
# Copy only necessary files for dependency installation to leverage Docker cache
COPY pyproject.toml uv.lock requirements.txt ./

# Install uv (if not already present in base image)
RUN pip install uv

# Install project dependencies
RUN uv sync

# Copy the rest of the application code
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the application
# Use 0.0.0.0 to make the server accessible from outside the container
# For production, consider using a more robust WSGI server like Gunicorn with Uvicorn workers
# GOOGLE_APPLICATION_CREDENTIALS and GAE_APPLICATION should ideally be handled via Docker secrets or mounted volumes
# and passed at runtime for security reasons.
# Example: docker run -e GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json" -e GAE_APPLICATION=true ...

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
