#Base Image
FROM python:3.9

#Workdir
WORKDIR /app

#Copy
COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

#RUN
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

#EXPOSE
EXPOSE 5000

#command
CMD ["gunicorn","-b","0.0.0.0:5000", "app:app"]

