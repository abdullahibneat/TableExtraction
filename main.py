import cv2
from matplotlib import pyplot as plt


def preProcess(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur image, kernel size determined by trial and error
    # Removes noise
    kernel_size = 7
    blur = cv2.blur(gray, (kernel_size, kernel_size))

    # Use adaptive thresholding to have only black and white pixels
    # Without adaptive shadows might black out regions in the image
    # Gaussian produces less noise compared to ADAPTIVE_THRESH_MEAN_C
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    return threshold


def main():
    # Read image
    img = cv2.imread("sample_table.jpg")
    img_copy = img.copy()

    # Process image
    processed = preProcess(img_copy)

    # Show image
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()
