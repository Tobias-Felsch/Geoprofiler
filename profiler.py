import cv2
import numpy as np

def find_black_line(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([180, 255, 16])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

def main():
    # Force a clean start
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Load the image
    image = cv2.imread('Auswahl_001.png')
    if image is None:
        print("Error: Could not load image.")
        return

    # Process the image
    mask, contours = find_black_line(image)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)

    # Display results
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', image_with_contours)

    # Wait for user input and close properly
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensures resources are released

if __name__ == "__main__":
    main()
