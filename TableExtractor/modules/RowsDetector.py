import cv2
import numpy as np

def findRows(cell_contours):
    rows = {}

    for cnt in cell_contours:
        # Approximate contour to a rectangle, get x, y, width and height
        x, y, width, height = cv2.boundingRect(cnt)

        # Ignore cell contours with width or height < 15px
        if width < 15 or height < 15:
            continue

        # Contour could have a strange shape, so replace original contour
        # with the approximated bounding rectange shape
        cnt = np.array([
            (x, y), # Top left
            (x + width, y), # Top right
            (x + width, y + height), # Bottom right
            (x, y + height) # Bottom left
        ]).reshape((4, 2))

        # Keep track of whether the contour has been assigned to a row
        added = False

        # Iterate over existing rows where:
        # row = y-coordinate of the row
        for row in rows.keys():
            # Add this cell to the row that is on the same line (i.e. y-axis ± 50px)
            # as this cell's contour
            if (row - 50) <= y <= (row + 50):
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