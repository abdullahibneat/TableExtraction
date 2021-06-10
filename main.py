import TableExtractor

#
# SIMPLE USAGE
#
# Pass the path to an image to the "extractTable" function, and return
# the tabular sctructure WITHOUT PERFORMING OCR (it uses cell numbers instead)

print(TableExtractor.extractTable("data/sample_table.jpg"))


#
# ADVANCED USAGE
#
# Pass an OCR function that takes in an OpenCV image and returns its text.
# 
# For instance, using tesserocr (UNCOMMENT __ALL__ THE LINES BELOW):

# from tesserocr import PyTessBaseAPI, PSM, OEM
# import re

# pattern = "[^a-zA-Z0-9:]"

# with PyTessBaseAPI(lang="eng", psm=PSM.SINGLE_BLOCK, oem=OEM.LSTM_ONLY) as api:
#     def ocrFunction(cell):
#         # Pass cell image to tesserocr
#         # More info: https://github.com/sirfz/tesserocr/issues/198#issuecomment-652572304
#         height, width = cell.shape
#         api.SetImageBytes(
#             imagedata=cell.tobytes(),
#             width=width,
#             height=height,
#             bytes_per_pixel=1,
#             bytes_per_line=width
#         )
#         text = api.GetUTF8Text().strip()
#         text = re.sub(pattern, "", text)
#         return text

#     print(TableExtractor.extractTable("data/sample_table.jpg", ocrFunction))