import cv2
from matplotlib import pyplot as plt
import numpy as np


def preProcess(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur image to remove noise
    # Determine kernel size by using image height and width
    height, width, _ = img.shape
    kernel_size = min(int(height * 0.0025), int(width * 0.0025))
    # Kernel must have odd values because of GaussianBlur
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = (kernel_size, kernel_size)
    print("kernel: " + str(kernel))
    blur = cv2.GaussianBlur(gray, kernel, 1)

    # Use adaptive thresholding to have only black and white pixels
    # Without adaptive shadows might black out regions in the image
    # Gaussian produces less noise compared to ADAPTIVE_THRESH_MEAN_C
    # Block size: above, both kernel values are odd, but block size must be even, therefore add 1.
    block_size = kernel_size * 2 + 1
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

    # Use laplacian to detect gradients in the image (i.e. lines)
    # This helps to improve table region detection in later stages
    laplacian = cv2.Laplacian(threshold, cv2.CV_64F)
    # Convert data type from 64f to unsigned 8-bit integer
    laplacian = np.uint8(np.absolute(laplacian))

    return (threshold, laplacian)


def findLargestQuadrilateralContour(contours, maxArea=None):
    maxAreaSet = maxArea is not None
    biggest_area = 0
    biggest_contour = None
    biggest_contour_approx = None
    for contour in contours:
        # Get the area of this contour
        area = cv2.contourArea(contour)

        # Reassign maxArea if it was originally None
        if not maxAreaSet:
            maxArea = area

        # Get the length of the perimeter
        perimeter = cv2.arcLength(contour, True)

        # Approximate a shape that resembles the contour
        # This is needed because the image might be warped, thus
        # edges are curved and not perfectly straight
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        # Check if area is bigger than previous contour but smaller than or equal to maxArea
        # and if the approximation contains only 4 sides (i.e. quadrilateral)
        if biggest_area < area <= maxArea and len(approx) == 4:
            biggest_area = area
            biggest_contour = contour
            biggest_contour_approx = approx
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


def findLines(img):
    height, width = img.shape
    
    erosion = cv2.erode(img, np.ones((3, 3)))

    horizontal_kernel = np.ones((1, width // 30))
    horizontal = cv2.dilate(erosion, horizontal_kernel)
    horizontal = cv2.erode(horizontal, horizontal_kernel)
    
    vertical_kernel = np.ones((height // 30, 1))
    vertical = cv2.dilate(erosion, vertical_kernel)
    vertical = cv2.erode(vertical, vertical_kernel)
    return cv2.bitwise_and(vertical, horizontal)


def main():
    # Read image
    img = cv2.imread("data/sample_table.jpg")
    img_copy = img.copy()
    height, width, _ = img.shape

    # Process image
    threshold, laplacian = preProcess(img_copy)

    # Find contours
    contours, _ = cv2.findContours(laplacian, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find table region
    # It is assumed the table takes up most of the image (less than 95%),
    # thus it can be identified by finding the largest contour with 4 sides
    maxArea = width * height * 0.95
    table_contour, table_contour_approx = findLargestQuadrilateralContour(contours, maxArea)
    table_pts, table_width, table_height = processContour(table_contour_approx[0])

    # Extract table region
    # Use warp to extract the table region from the processed image
    # by mapping table points to a new image of size table_width x table_height
    target_points = np.float32([[0, 0], [table_width, 0], [table_width, table_height], [0, table_height]])
    matrix = cv2.getPerspectiveTransform(table_pts, target_points)
    # Apply warp to threshold image
    warped = cv2.warpPerspective(threshold, matrix, (table_width, table_height))

    # Find horizontal and vertical lines
    lines = findLines(warped)

    # FOR DEBUG PURPOSES ONLY
    images = [(img, "original"), (threshold, "threshold"), (laplacian, "laplacian")]

    # Create new image with table contour displayed on top of processed image
    table_contour_image = cv2.cvtColor(laplacian.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(table_contour_image, table_contour, -1, (0, 0, 255), 10)  # Contour
    cv2.drawContours(table_contour_image, table_contour_approx, -1, (0, 255, 0), 10)  # Approximation
    images.append((table_contour_image, "contour"))
    images.append((warped, "warped"))
    images.append((lines, "lines"))

    # Show images
    for image, title in images:
        plt.figure(title)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()
