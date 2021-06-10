# Table Extraction

![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/abdullahibneat/table-extraction)

![Extracting tabular data as JSON data](https://i.imgur.com/vUUQ4g1.png)

![The web interface](https://i.imgur.com/on76ccg.png)

## Overview

This framework was developed as part of my undergraduate final year project at University and allows for the extraction of tabular data from raster images. It uses **line information** to locate cells, and an algorithm arranges the cells in memory to reconstruct the tabular structure. It then uses the Tesseract OCR engine to extract the text and returns the entire table as JSON data.  It achieved 89% cell detection accuracy when extracting prayer times from timetables (see `data` folder for some examples). 

The main drawbacks are as follows:

 - Heavily relies on ruling lines. The table must have all column and row separators, and blurry images can cause a drop in line detection
 - Table region detection is quite rudimentary: it looks for the largest quadrilateral in the image
 - It can only detect one table
 - Tesseract needs more fine tuning for better OCR processing, as sometimes text is not recognized properly.

Below is a summary of how the framework works. This structure is reflected in `TableExtractor.py`.

![Overview of processes involved](https://i.imgur.com/oz6YSGK.jpg)

## Docker

This is the recommended way to run this project as the environment is all set up and ready to use. For convenience, Docker images are automatically built and released on [Docker Hub](https://hub.docker.com/repository/docker/abdullahibneat/table-extraction).

To run the Docker container locally:

```
docker pull abdullahibneat/table-extraction
docker run -d -p 5000:5000 abdullahibneat/table-extraction
```

Then visit http://localhost:5000 and you're ready to go!

When using a cloud provider, you can change the port by setting the `PORT` environment variable. In Heroku, the port is set automatically so this repository can simply be pushed to the Heroku remote.

## Manual setup

### OCR setup

An OCR engine is NOT required to run the project, though without one the returned table object will return cell numbers instead of the cell contents.

This project uses [tesserocr](https://github.com/sirfz/tesserocr) as the Tesseract wrapper out-of-the-box. Follow the instructions there to set up tesserocr.

Alternatively, use your own OCR implementation by removing the tesserocr requirement from `requirements.txt` and updating the code in `main.py` and/or `server.py` with your own implementation.

### Get started

1. Make sure Python 3.7.x is installed. `❗❗❗THIS IS IMPORTANT❗❗❗`
2. `Recommended:` Set up a Python 3.7 [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
3. Install the requirements (tesserocr might require extra steps, see below): `pip install -r requirements.txt`
4. Run the `main.py` file

### Flask API server

A simple Flask API was written to interact with the table extractor. Run the `app` module with Flask:

```
FLASK_APP=app flask run
```

and visit the address (default: `http://localhost:5000`). Alternatively, send the image as form data (it can have any name) in a `POST` request to the root endpoint:

```
curl -F image=@myImage.jpg http://localhost:5000
```