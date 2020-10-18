FROM python:3.8.5-alpine3.12

RUN apk --update add --no-cache g++

RUN pip install pandas

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY ./app /app

WORKDIR /app

CMD [ "python", "app.py" ]