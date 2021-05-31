# Table Extraction

## Tesseract setup

Follow the instruction from [https://github.com/sirfz/tesserocr](https://github.com/sirfz/tesserocr).

## Get started

1. Make sure Python 3.7.x is installed
2. Set up a Python 3.7 [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
3. Install the requirements (tesserocr might require extra steps, see below): `pip install -r requirements.txt`
4. Run the `main.py` file

## Flask API server

A simple Flask API was written to interact with the table extractor. Run the `server.py` file with Flask:

```
FLASK_APP=server
flask run
```

and visit the address (default: `127.0.0.1:5000`). Alternatively, store the image as form data (it can have any name) and send a `POST` request to the root endpoint.