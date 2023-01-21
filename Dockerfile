FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install spacy ftfy
RUN python -m spacy download en

RUN pip install matplotlib
RUN pip install accelerate
RUN pip install ImageHash
# Upgrading all python packages to latest version in one line
RUN pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U




