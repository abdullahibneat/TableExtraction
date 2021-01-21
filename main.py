import cv2
from matplotlib import pyplot as plt
import numpy as np
from random import randint, sample
from tesserocr import PyTessBaseAPI, PSM, OEM
from PIL import Image


def preProcess(img):
    # Blur image to remove noise
    # Determine kernel size by using image height and width
    height, width = img.shape
    kernel_size = max(int(height * 0.005), int(width * 0.005))
    # Kernel must have odd values because of GaussianBlur
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = (kernel_size, kernel_size)
    print("kernel: " + str(kernel))
    blur = cv2.GaussianBlur(img, kernel, 1)

    # Use adaptive thresholding to have only black and white pixels
    # Without adaptive shadows might black out regions in the image
    # Gaussian produces less noise compared to ADAPTIVE_THRESH_MEAN_C
    # Block size: above, both kernel values are odd, but block size must be even, therefore add 1.
    block_size = kernel_size * 2 - 1
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

    # Use laplacian to detect gradients in the image (i.e. lines)
    # This helps to improve table region detection in later stages
    laplacian = cv2.Laplacian(threshold, cv2.CV_64F)
    # Convert data type from 64f to unsigned 8-bit integer
    laplacian = np.uint8(np.absolute(laplacian))

    return laplacian


def findLargestQuadrilateralContour(contours):
    # Sort contours from smallest area to biggest
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    biggest_contour = None
    biggest_contour_approx = None

    for cnt in sorted_contours:
        # Get the length of the perimeter
        perimeter = cv2.arcLength(cnt, True)

        # Approximate a shape that resembles the contour
        # This is needed because the image might be warped, thus
        # edges are curved and not perfectly straight
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)

        # Check if the approximation contains only 4 sides
        # (i.e. quadrilateral)
        if len(approx) == 4:
            biggest_contour = cnt
            biggest_contour_approx = approx
            break

    return [biggest_contour], [biggest_contour_approx]


def processContour(approx):
    # Reshape array([x, y], ...) to array( array([x], [y]), ...)
    approx = approx.reshape((4, 2))

    # Sort points in clockwise order, starting from top left
    pts = np.zeros((4, 2), dtype=np.float32)

    # Add up all values
    # Smallest sum = top left point
    # Largest sum = bottom right point
    s = approx.sum(axis=1)
    pts[0] = approx[np.argmin(s)]
    pts[2] = approx[np.argmax(s)]

    # For the other 2 points, compute difference between all points
    # Smallest difference = top right point
    # Largest difference = bottom left point
    diff = np.diff(approx, axis=1)
    pts[1] = approx[np.argmin(diff)]
    pts[3] = approx[np.argmax(diff)]

    # Calculate smallest height and width
    width = int(min(pts[1][0] - pts[0][0], pts[2][0] - pts[3][0]))
    height = int(min(pts[3][1] - pts[0][1], pts[2][1] - pts[1][1]))

    return pts, width, height


# This funciton takes a line image as an input, and removes small lines
# (considered as noise) by finding all contours, and fill small contours
# with white
def removeNoisyLines(lines):
    # Binarize the lines image
    _, cleanedImage = cv2.threshold(lines, 250, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2))

    # Erode the image to thicken lines
    cleanedImage = cv2.erode(lines, kernel, iterations=2)

    # Find all contours in the image
    contours, _ = cv2.findContours(cleanedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 275:
            # Fill small contours with white
            cv2.drawContours(cleanedImage, [cnt], -1, (255,255,255), -1)

    # Previously lines were thickened, restore them to their original weight
    cleanedImage = cv2.dilate(cleanedImage, kernel, iterations=2)

    return cleanedImage


def findLinesAndIntersections(img):
    # Adapted from https://docs.opencv.org/4.4.0/dd/dd7/tutorial_morph_lines_detection.html

    # Get image height and width to dynamically change
    # horizontal and vertical kernel sizes
    height, width = img.shape

    # To find horizontal lines, run a horizontal kernel (e.g. [1 1 1 1])
    # Dilation finds lines, but shrinks their lengths, so
    # follow with Erosion to restore original lines' size
    horizontal_kernel = np.ones((1, width // 100))
    horizontal = cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)
    horizontal = removeNoisyLines(horizontal)
    
    # To find vertical lines, run a vertical kernel
    vertical_kernel = np.ones((height // 100, 1))
    vertical = cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)
    vertical = removeNoisyLines(vertical)

    lines = cv2.bitwise_and(vertical, horizontal)
    lines = cv2.erode(lines, np.ones((3, 3)), iterations=3)

    # Binarize the image, values less than 250 (almsot full white) are
    # treated as black pixels, improving cell contour detection.
    # Without this, grey lines are sometimes ignored by cv2.findContours
    _, lines = cv2.threshold(lines, 250, 255, cv2.THRESH_BINARY)

    return lines


def extractRows(cell_contours):
    # Get a subset of the cell contours (10%) and compute an average cell height
    sample_cells = sample(cell_contours, int(len(cell_contours) * 0.1))
    avg_height = sum([cv2.boundingRect(cnt)[3] for cnt in sample_cells]) // len(sample_cells)
    print("Average cell height: " + str(avg_height))

    rows = {}

    for cnt in cell_contours:
        # Approximate contour to a rectangle, get x, y, width and height
        x, y, width, height = cv2.boundingRect(cnt)

        # Contour could have a strange shape, so replace original contour
        # with the approximated bounding rectange shape
        cnt = np.array([
            (x, y), # Top left
            (x + width, y), # Top right
            (x + width, y + height), # Bottom right
            (x, y + height) # Bottom left
        ]).reshape((4, 2))

        # x, y are coordinates of the top-left point, get the center of rectangle
        y = y + int(height / 2)

        # Keep track of whether the contour has been assigned to a row
        added = False

        # Iterate over existing rows where:
        # row = y-coordinate of the row
        for row in rows.keys():
            # Add this contour to the row that is within a margin of error
            # (± avg cell height)
            # This simple algorithm works well because of the table warping,
            # meanining all rows should be horizontally parallel to each other.
            if (row - avg_height) <= y <= (row + avg_height):
                rows[row].append(cnt)
                added = True
                break

        # If the row wasn't added, create a new row with this cell's y-coordinate
        # as the row
        if not added:
            rows[y] = [cnt]

    # Sort rows top to bottom.
    rows = dict(sorted(rows.items()))

    # Sort cells left to right
    for key, value in rows.items():
        rows[key] = sorted(value, key=lambda cnt: cv2.boundingRect(cnt)[0])
    
    return rows


# This funciton is used to recursively find the leaf entries in a dictionary, and replace
# the last list values with dictionaries. This is done beacuse a new table heading should
# be represented by a dictionary type, whereas column values are stored in a list.
def leafListToDict(column):
    # If the column is a list...
    # This could happen if a heading has multiple values followed by a column split
    # E.g.
    #       +-----------------+------------------+-------------------+
    #       |        A        |        B         |         C         |
    #       +-----------------+------------------+-------------------+
    #       |     value1      |     value2       |      value3       |
    #       +-----------------+------------------+-------------------+
    #       |     value4      |     value5       |      value6       |
    #       +--------+--------+--------+---------+---------+---------+
    #       |   D    |   E    |   F    |    G    |    H    |    I    |
    #       +--------+--------+--------+---------+---------+---------+
    #       | value7 | value8 | value9 | value10 | value11 | value12 |
    #       +--------+--------+--------+---------+---------+---------+
    # Column A has 2 values (value1, value4) followed by a column split (D, E)
    if type(column) is list:
        # If the last item in the list is a dictionary, iterate over that dictionary to
        # find the leaf list
        if type(column[-1]) is dict:
            return leafListToDict(column[-1])
        
        # Otherwise create a new dictiorary and return it
        new_value = {}
        column.append(new_value)
        return [new_value]

    # If the values are all empty lists...
    # any(list) returns True if list contains non-empty lists
    # E.g. any([[1], [2], [3]]) = True, any([[], [], []]) = False
    if not any(column.values()):
        # ...replace them with dictionaries
        for key in column:
            column[key] = {}
        return column.values()
    # Otherwise recursively iterate all the dictionaries until the leaf key-value pair is
    # reached. Double for-loop is used to flatten the return array
    # E.g. [[a], [b], [c]] => [a, b, c]
    return [column for child in column.values() for column in leafListToDict(child)]


def reconstructTable(rows, warped):
    # Reconstruct the table following a top-to-bottom approach. Iterate over each row
    # and check for the number of cells. If there are more cells than the previous row,
    # this will indicate columns have been split, and the new row is treated as a new 
    # heading in the table.
    # Otherwise, if the same number of cells appear, this means the row contains new
    # values for the previous column, so add the cell content to the existing column.

    # EXAMPLE TABLE:
    # +-------+-------+
    # |   A   |   B   |
    # +---+---+---+---+
    # | C | D | E | F |
    # +---+---+---+---+
    # | 1 | 2 | 3 | 4 |
    # +---+---+---+---+

    # Store table as a disctionary, where:
    #   key = column name
    #   value = list of cell values
    # In the example above, table will look like the following: 
    #
    # table = {
    #   A: {
    #       C: [1],
    #       D: [2],
    #   },
    #   B: {
    #       E: [3],
    #       F: [4]
    #   }
    # }
    table = {}

    # Columns is a reference to the values of the heading names.
    # For example, in the above example:
    # after the first iteration:    columns = [[], []]
    # after the second iteration:   columns = [[], [], [], []]
    # after the third iteration:    columns = [[1], [2], [3], [4]]
    columns = None
    columns_sizes = [] # Keep track of column sizes

    # Sometimes Tessearct returns an empty string when parsing a cell. Keep track
    # of cell number to replace empty strings.
    cell_number = 0

    with PyTessBaseAPI(lang="eng", psm=PSM.SINGLE_BLOCK, oem=OEM.LSTM_ONLY) as api:
        for cells in rows.values():
            cell_sizes = [] # Keep track of cell sizes

            # Extract cell text (will be replaced with OCR)
            cell_contents = []
            for cnt in cells:
                # Extract cell region from image
                x1, y1 = cnt[0]
                x2, y2 = cnt[2]
                cell_sizes.append(x2 - x1) # Add cell width to the list of cell sizes
                cell = warped[y1:y2, x1:x2]
                # Convert from cv2 to PIL
                cell = Image.fromarray(cell)
                # Parse image in Tesseract
                api.SetImage(cell)
                text = api.GetUTF8Text().strip()
                if text == "":
                    text = "(failed) cell #" + str(cell_number)
                cell_contents.append(text)
                cell_number += 1
            
            if columns is None:
                # FIRST ITERATION
                # Add first row to the table
                for cell in cell_contents:
                    table[cell] = []
                columns = list(table.values())
                columns_sizes = cell_sizes

            elif len(cell_contents) > len(columns):
                # DIFFERENT NUMBER OF CELLS
                # Columns have been split, add this row as new headings

                # Replace the previous columns to be new dictionaries.
                # At this line, columns contains the lists of the last headings. Because new
                # headings have been found, the last headings are converted from lists to
                # dictionaries.
                columns = leafListToDict(table)

                # Keep track of the previous headings
                previous_headings = list(columns)

                # Create new columns for each of the new headings
                columns = [[] for _ in cell_contents]

                # Split the new headings into lists of equal size.
                # For instance, in the example table above they are split in groups of 2:
                #   - [C, D] are children of A
                #   - [E, F] are children of B
                # This is done by comparing the current column size (+1%) to the cell sizes.
                # When the column size is exceeded, this will indicate a new column has started.
                current_column_index = 0 # Keep track of current column
                current_width = 0 # Aggregate all cell sizes to check against column size

                for i, heading in enumerate(cell_contents):
                    current_width += cell_sizes[i] # Add current cell width to current column

                    # If cell doesn't fit in the current column, move to the next one
                    if current_width > columns_sizes[current_column_index] * 1.01:
                        current_column_index += 1
                        current_width = cell_sizes[i]

                    # Add this new heading as child of previous heading
                    previous_headings[current_column_index][heading] = columns[i]

                # Reset columns sizes
                columns_sizes = cell_sizes

            elif len(cell_contents) == len(columns):
                # SAME NUMBER OF CELLS
                # add all cells one by one to each column
                for i in range(len(cells)):
                    columns[i].append(cell_contents[i])
    return table


def main():
    # READ IMAGE
    img = cv2.imread("data/sample_table.jpg", 0)
    img_copy = img.copy()

    # PROCESS IMAGE
    laplacian = preProcess(img_copy)

    # FIND CONTOUR
    contours, _ = cv2.findContours(laplacian, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # FIND TABLE REGION
    # It is assumed the table takes up most of the image,
    # thus it can be identified by finding the largest contour with 4 sides
    table_contour, table_contour_approx = findLargestQuadrilateralContour(contours)
    table_pts, table_width, table_height = processContour(table_contour_approx[0])

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
    scale_factor = 750 / table_width
    warped = cv2.resize(warped, (0, 0), fx=scale_factor, fy=scale_factor)
    warped_mask = cv2.resize(warped_mask, (0, 0), fx=scale_factor, fy=scale_factor)
    # Apply threshold
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)

    # FIND HORIZONTAL & VERTICAL LINES
    # Find horizontal and vertical lines
    lines = findLinesAndIntersections(warped)
    # Since the funciton above might get rid of the black area outside the table
    # region, apply mask again
    lines = cv2.bitwise_and(lines, warped_mask)

    # EXTRACT CELLS
    # Get each cell's contour
    cell_contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove cell contours with width < 15px (table is 750px wide, 15px = 2%)
    cell_contours = list(filter(lambda x: cv2.boundingRect(x)[2] > 15, cell_contours))

    # Group cells by row
    rows = extractRows(cell_contours)
    print("Found " + str(len(rows.values())) + " rows, " + str(sum([len(c) for c in rows.values()])) + " cells")

    # CREATE TABLE IMAGE WITHOUT LINES
    # This will help the OCR engine perform better.
    # Start with a full white image, add the cells as black rectangles and use the OR operation
    # to remove all the lines.
    text_mask = np.full(warped.shape, 255).astype(warped.dtype)
    text_mask = cv2.drawContours(text_mask, cell_contours, -1, (0, 0, 0), -1)
    text_only = cv2.bitwise_or(warped, text_mask)

    # Apply a blur to improve Tesseract's accuracy
    text_only = cv2.medianBlur(text_only, 3)

    # RECONSTRUCT TABLE STRUCTURE
    table = reconstructTable(rows, text_only)
    print(table)

    # FOR DEBUG PURPOSES ONLY
    images = [(img, "original"), (laplacian, "laplacian")]

    # Create new image with table contour displayed on top of processed image
    table_contour_image = cv2.cvtColor(laplacian.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(table_contour_image, table_contour, -1, (0, 0, 255), 10)  # Contour
    cv2.drawContours(table_contour_image, table_contour_approx, -1, (0, 255, 0), 10)  # Approximation
    images.append((table_contour_image, "contour"))
    images.append((table_img, "table"))
    images.append((warped, "warped"))
    images.append((lines, "lines"))
    # Create new image to display cell contours
    cell_contours_image = lines.copy()
    cell_contours_image = cv2.cvtColor(cell_contours_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(cell_contours_image, cell_contours, -1, (0, 0, 255), 2)
    # Add overlay showing contour index in image
    for i, cnt in enumerate(cell_contours):
        # Get contour coordinates
        # Refer to https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#moments
        M = cv2.moments(cnt)
        coordinates = (int(M['m10']/(M['m00'] + 1)), int(M['m01']/(M['m00'] + 1)))
        # Put text with index contour index at above coordinates
        cv2.putText(cell_contours_image, str(i), coordinates, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 2)
    images.append((cell_contours_image, "cell contours"))
    # Create new image to display detected rows
    rows_img = warped.copy()
    rows_img = cv2.cvtColor(rows_img, cv2.COLOR_GRAY2BGR)
    # Colour-coordinate cells based on row and display cell contour index
    for _, value in rows.items():
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.drawContours(rows_img, value, -1, color, 2)
        for i, cnt in enumerate(value):
            M = cv2.moments(cnt)
            coordinates = (int(M['m10']/(M['m00'] + 1)), int(M['m01']/(M['m00'] + 1)))
            cv2.putText(rows_img, str(i), coordinates, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
    images.append((rows_img, "rows"))

    # Show images
    for image, title in images:
        plt.figure(title)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()
