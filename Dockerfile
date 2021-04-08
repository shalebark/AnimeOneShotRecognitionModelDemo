FROM python:3.7-slim

RUN apt-get update && apt-get upgrade
RUN apt-get install -y build-essential cmake python3-opencv

WORKDIR /code

COPY . .

RUN pip install -r requirements.txt

WORKDIR /app

ENTRYPOINT  [ "python", "/code/demo.py" ]

