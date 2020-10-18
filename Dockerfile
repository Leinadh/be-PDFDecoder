FROM python:3.8.5

RUN pip install --no-cache-dir matplotlib pandas

RUN apt-get update
RUN apt-get install poppler-utils -y

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY ./app /app

WORKDIR /app

CMD [ "python", "app.py" ]