# Base Image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install dependencies in a single layer and remove unnecessary files to reduce image size
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     python3-dev \
#     libpq-dev \
#     && pip install --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt \
#     && python -m nltk.downloader stopwords wordnet \
#     && rm -rf /var/lib/apt/lists/*  # Clean up APT cache to reduce image size

RUN pip install -r requirements.txt && \ 
    python -m nltk.downloader stopwords wordnet 

    # Clean up APT cache to reduce image size

# Copy application files
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Expose the application port
EXPOSE 5000

# Set the default command to run the app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
