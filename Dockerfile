# 1. Start with an official, lightweight Python base operating system layer
FROM python:3.11-slim

# 2. Set the internal working directory inside the container
WORKDIR /app

# 3. Copy our python requirements list into the container first
# (This allows Docker to cache our library installations for fast builds)
RUN pip install --no-cache-dir fastapi uvicorn qdrant-client pydantic

# 4. Copy all our local project files into the container's /app directory
COPY app.py /app/app.py

# 5. Inform Docker that the container will listen for network traffic on port 8000
EXPOSE 8000

# 6. Define the exact command to run when the container starts up live
# We set host to 0.0.0.0 so it accepts incoming web connections from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]