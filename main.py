import cv2
from matplotlib import pyplot as plt


def preProcess(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur image to remove noise
    # Determine kernel size by using image height and width
    height, width, _ = img.shape
    blur_kernel = [int(height * 0.0025), int(width * 0.0025)]
    # Kernel must have odd values because of GaussianBlur
    if blur_kernel[0] % 2 == 0:
        blur_kernel[0] += 1
    if blur_kernel[1] % 2 == 0:
        blur_kernel[1] += 1
    blur_kernel = (blur_kernel[0], blur_kernel[1])
    print("kernel: " + str(blur_kernel))
    blur = cv2.GaussianBlur(gray, blur_kernel, 1)

    # Use adaptive thresholding to have only black and white pixels
    # Without adaptive shadows might black out regions in the image
    # Gaussian produces less noise compared to ADAPTIVE_THRESH_MEAN_C
    # Block size: above, both kernel values are odd, but block size must be even, therefore add 1.
    block_size = blur_kernel[0] + blur_kernel[1] + 1
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
    return threshold


def findLargestQuadrilateralContour(contours, maxArea=None):
    maxAreaSet = maxArea is not None
    biggest_area = 0
    biggest_contour = None
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
    return [biggest_contour]


def main():
    # Read image
    img = cv2.imread("sample_table.jpg")
    img_copy = img.copy()
    height, width, _ = img.shape

    # Process image
    processed = preProcess(img_copy)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find table region
    # It is assumed the table takes up most of the image (less than 95%),
    # thus it can be identified by finding the largest contour with 4 sides
    maxArea = width * height * 0.95
    table_contour = findLargestQuadrilateralContour(contours, maxArea)

    # FOR DEBUG PURPOSES ONLY
    # Create new image with table contour displayed on top of processed image
    table_contour_image = cv2.cvtColor(processed.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(table_contour_image, table_contour, -1, (0, 0, 255), 10)

    # Show image
    plt.imshow(cv2.cvtColor(table_contour_image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()
