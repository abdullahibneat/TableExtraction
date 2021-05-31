from random import sample
import cv2
import numpy as np

def findRows(cell_contours):
    # Get a subset of the cell contours (10%) and compute an average cell height
    sample_cells = sample(cell_contours, int(len(cell_contours) * 0.1))
    avg_height = sum([cv2.boundingRect(cnt)[3] for cnt in sample_cells]) // len(sample_cells)

    rows = {}

    for cnt in cell_contours:
        # Approximate contour to a rectangle, get x, y, width and height
        x, y, width, height = cv2.boundingRect(cnt)

        # Ignore cell contours with width < 8px (table is 1500px wide, 8px = 0.5%)
        # or height less than 75% of the average height
        if width < 8 or height < avg_height * 0.75:
            continue

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