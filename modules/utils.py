import cv2
import numpy as np

# Function to find the largest 4-sided contour from an array of countours
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


# Function to sort points in a contour in clockwise order, starting from top left
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
