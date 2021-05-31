from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from time import time
import TableExtractor
from os.path import exists
from os import remove, makedirs

app = Flask(__name__)

# Restrict file upload to these extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
# Limit upload size to 16MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
# "jsonify" sorts JSON keys alphabetically by default
# This turns off this behaviour to preserve table columns' order
app.config['JSON_SORT_KEYS'] = False

# Function to check that uploaded files are of image type
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["POST"])
def index():
    """
    This POST endpoint expects an image to be sent within form data. The upload field can have any name.

    E.g.
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" />
    </form>

    Note: the "enctype" is important if using an HTML form!
    """
    if(len(request.files) == 0):
        return jsonify(error="You must submit an image as form data.")

    # Get the first file from the form data
    file = request.files[next(iter(request.files))]
    filename = secure_filename(file.filename)

    # Check that a file was actually sent. Empty forms send an empty file.
    if(not file or filename == ""):
        return jsonify(error="You must submit an image as form data.")
    
    # Check file is an image
    if(not allowed_file(filename)):
        return jsonify(error="The image must be any of the following formats: " + ", ".join(ALLOWED_EXTENSIONS))

    try:
        if(not exists("./tmp")):
            makedirs("tmp")
            
        # Save file in ./tmp directory
        tmp_filename = str(time()) + filename
        tmp_path = "./tmp/" + tmp_filename
        file.save(tmp_path)

        # Perform table extraction
        tableData = TableExtractor.extractTable(tmp_path)
    except:
        tableData = jsonify(error="An error occurred while parsing the image. Try again with a clearer picture.")
    finally:
        # Delete temporary file
        if(exists(tmp_path)):
            remove(tmp_path)
    
    return tableData