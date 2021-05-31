import cv2
import numpy as np

def findLines(img):
    # Adapted from https://docs.opencv.org/4.4.0/dd/dd7/tutorial_morph_lines_detection.html

    # Get image height and width to dynamically change
    # horizontal and vertical kernel sizes
    height, width = img.shape

    # Erode image to thicken lines
    eroded = cv2.erode(img, np.ones((3, 3)))

    kernel_length = 3 / 100

    # To find horizontal lines, run a horizontal kernel (e.g. [1 1 1 1])
    # Dilation finds lines, but shrinks their lengths, so
    # follow with Erosion to restore original lines' size
    horizontal_kernel = np.ones((1, int(width * kernel_length)))
    horizontal = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, horizontal_kernel)
    
    # To find vertical lines, run a vertical kernel
    vertical_kernel = np.ones((int(height * kernel_length), 1))
    vertical = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, vertical_kernel)

    lines = cv2.bitwise_and(vertical, horizontal)
    lines = cv2.erode(lines, np.ones((3, 3)), iterations=3)

    return lines