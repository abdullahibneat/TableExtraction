import cv2
import numpy as np

def process(img):
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