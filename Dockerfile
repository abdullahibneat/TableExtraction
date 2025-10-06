FROM python:slim-bullseye

RUN apt-get -y update \
    && apt-get -y upgrade \
    # tesserocr requirements
    && apt-get -y install tesseract-ocr libtesseract-dev libleptonica-dev

# Required for tesserocr:
# https://github.com/sirfz/tesserocr/issues/165#issuecomment-445789709
ENV LC_ALL=C
# Use port 5000 by default, could be overwritten by cloud providers (e.g. Heroku)
ENV PORT=5000

EXPOSE ${PORT}

WORKDIR /app

COPY . .

# Build tesserocr wheel and install dependancies
RUN apt-get -y install pkg-config build-essential \
    # Use piwheels for arm builds
    && pip3 install -r requirements-docker.txt --extra-index-url https://www.piwheels.org/simple \
    # Remove build dependencies
    && apt-get -y purge --auto-remove pkg-config build-essential

# Run flask app
CMD gunicorn -b 0.0.0.0:$PORT -w 4 app:app --timeout 120
