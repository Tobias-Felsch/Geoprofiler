import cv2
import numpy as np


def find_black_line(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for black color (adjust these values based on your image)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([180, 255, 16])

    # Create a mask to extract the black color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def main():
    # Load the image
    image = cv2.imread('simple_geoprofile.jpg')  # Replace 'your_image.jpg' with the actual image file

    # Find black line
    black_line_contours = find_black_line(image)

    # Draw contours of the black line on the original image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, black_line_contours, -1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Result', image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()