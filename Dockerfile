FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install spacy ftfy
RUN python -m spacy download en


# Upgrading all python packages to latest version in one line
RUN pip install --upgrade pip && pip list --outdated --format=freeze | cut -d = -f 1 | xargs -n1 pip install -U




