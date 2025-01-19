#trying to optimize code without multistage build
#Base Image
FROM python:3.9-slim 

#Workdir
WORKDIR /app

# Install system dependencies for building certain Python packages (gcc, build-essential, python3-dev)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

#copy only the requirements.txt file first (to use docker caching)
COPY requirements.txt .

# Upgrade pip to avoid issues with older versions
RUN pip install --upgrade pip
#install dependencies and remove cache after installation (with less no of layers)
# Install dependencies
RUN pip install -v -r requirements.txt

# Download NLTK data (split into separate RUN to isolate issues)
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

# Clean up the pip cache
RUN rm -rf /root/.cache/pip

#Copy
COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl


#EXPOSE
EXPOSE 5000

#command
CMD ["gunicorn","-b","0.0.0.0:5000", "app:app"]

