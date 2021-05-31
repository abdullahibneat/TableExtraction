from modules import PreProcessing, utils, LinesDetector, RowsDetector, TableBuilder
import cv2
import numpy as np

def extractTable(imgPath):
    # Dictonary to store data to be returned
    ret = {}

    img = cv2.imread(imgPath, 0)

    # PROCESS IMAGE
    laplacian = PreProcessing.process(img)

    # FIND CONTOUR
    contours, _ = cv2.findContours(laplacian, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # FIND TABLE REGION
    # It is assumed the table takes up most of the image,
    # thus it can be identified by finding the largest contour with 4 sides
    table_contour, table_contour_approx = utils.findLargestQuadrilateralContour(contours)
    table_pts, table_width, table_height = utils.processContour(table_contour_approx[0])

    # EXTRACT TABLE REGION
    # Start with a full black image
    mask = np.zeros(img.shape).astype(img.dtype)
    # Create a mask for the table region
    cv2.fillPoly(mask, table_contour, (255, 255, 255))
    # Apply the mask to the thresholded image, filling the region
    # outside of the table with white
    table_img = cv2.bitwise_and(img, mask)

    # WARP TABLE
    # Use warp to extract the table region from the processed image
    # by mapping table points to a new image of size table_width x table_height
    target_points = np.float32([[0, 0], [table_width, 0], [table_width, table_height], [0, table_height]])
    matrix = cv2.getPerspectiveTransform(table_pts, target_points)
    # Apply warp to the image to extract the tbale region
    warped = cv2.warpPerspective(table_img, matrix, (table_width, table_height))
    # Apply warp to mask
    warped_mask = cv2.warpPerspective(mask, matrix, (table_width, table_height))
    # Resize warped and mask to have width 750px
    scale_factor = 1500 / table_width
    warped = cv2.resize(warped, (0, 0), fx=scale_factor, fy=scale_factor)
    warped_mask = cv2.resize(warped_mask, (0, 0), fx=scale_factor, fy=scale_factor)
    warped = cv2.GaussianBlur(warped, (5, 5), 2)
    # Apply threshold
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)

    # FIND HORIZONTAL & VERTICAL LINES
    lines = LinesDetector.findLines(warped)
    # Since the funciton above might get rid of the black area outside the table
    # region, apply mask again
    lines = cv2.bitwise_and(lines, warped_mask)

    # EXTRACT CELLS
    # Get each cell's contour
    cell_contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Group cells by row
    # extractRows returns a Python dictonary with
    #   key = y value of the row
    #   value = array of cell contours in the row
    rows = RowsDetector.findRows(cell_contours)

    # Compute number of rows and cells
    ret["rows"] = len(rows.values())
    ret["cells"] = sum([len(c) for c in rows.values()])

    # CREATE TABLE IMAGE WITHOUT LINES
    # This will help the OCR engine perform better.
    # Start with a full white image, add the cells as black rectangles and use the OR operation
    # to remove all the lines.
    text_mask = np.full(warped.shape, 255).astype(warped.dtype)
    # Merge (sum()) all the cell contours from rows (key-value dictonary)
    text_mask = cv2.drawContours(text_mask, sum(rows.values(), []), -1, (0, 0, 0), -1)
    # Use close operation to dilate and erode image reducing overall noise
    text_only = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, np.ones((3,3)))
    text_only = cv2.bitwise_or(warped, text_mask)

    # RECONSTRUCT TABLE STRUCTURE
    ret["table"] = TableBuilder.reconstructTable(rows, text_only)

    return ret