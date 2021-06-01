# Table Extraction

![Extracting tabular data as JSON data](https://i.imgur.com/vUUQ4g1.png)

## Overview

This framework was developed as part of my undergraduate final year project at University and allows for the extraction of tabular data from raster images. It uses **line information** to locate cells, and an algorithm arranges the cells in memory to reconstruct the tabular structure. It then uses the Tesseract OCR engine to extract the text and returns the entire table as JSON data.  It achieved 89% cell detection accuracy when extracting prayer times from timetables (see `data` folder for some examples). 

The main drawbacks are as follows:

 - Heavily relies on ruling lines. The table must have all column and row separators, and blurry images can cause a drop in line detection
 - Table region detection is quite rudimentary: it looks for the largest quadrilateral in the image
 - It can only detect one table
 - Tesseract needs more fine tuning for better OCR processing, as sometimes text is not recognized properly.

Below is a summary of how the framework works. This structure is reflected in `TableExtractor.py`.

![Overview of processes involved](https://i.imgur.com/oz6YSGK.jpg)

## Tesseract setup

Follow the instruction from [https://github.com/sirfz/tesserocr](https://github.com/sirfz/tesserocr).

## Get started

1. Make sure Python 3.7.x is installed. `❗❗❗THIS IS IMPORTANT❗❗❗`
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

In production, use Gunicorn:

```
gunicorn server:app
```