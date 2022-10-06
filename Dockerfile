FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install spacy ftfy
RUN python -m spacy download en

RUN pip install --upgrade diffusers
