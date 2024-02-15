FROM python:3.9-slim

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx tesseract-ocr-khm libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV NAME ocr_service
WORKDIR /ocr_service


COPY . /ocr_service


RUN pip3 install -r requirements.txt

EXPOSE 8001