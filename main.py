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
# from TableExtractor import getOCRFunction

# with PyTessBaseAPI(lang="eng", psm=PSM.SINGLE_BLOCK, oem=OEM.LSTM_ONLY) as api:
#     print(TableExtractor.extractTable("data/sample_table.jpg", getOCRFunction(api)))