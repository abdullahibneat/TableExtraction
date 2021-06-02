FROM python:3.7-buster

RUN apt-get -y update
RUN apt-get -y upgrade
# tesserocr requirements
RUN apt-get -y install tesseract-ocr libtesseract-dev libleptonica-dev pkg-config

# Required for tesserocr:
# https://github.com/sirfz/tesserocr/issues/165#issuecomment-445789709
ENV LC_ALL=C
# Use port 5000 by default, could be overwritten by cloud providers (e.g. Heroku)
ENV PORT=5000

EXPOSE ${PORT}

WORKDIR /app

COPY . .

RUN pip3 install -r requirements-docker.txt

# Get the port from $PORT environment variable
CMD gunicorn -b 0.0.0.0:$PORT server:app