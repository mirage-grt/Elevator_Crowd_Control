import cv2
import numpy as np
import os

def capture_image():
    if not os.path.exists("sample_images"):
        os.makedirs("sample_images")
    
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    green = (0, 255, 0)
    cv2.rectangle(img, (100, 100), (400, 400), green, -1)
    img_path = "sample_images/elevator.jpg"
    cv2.imwrite(img_path, img)
    return img_path

def detect_occupancy(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]

    green_ratio = green_pixels / total_pixels

    return "FULL" if green_ratio < 0.2 else "NOT FULL"

def main():
    image_path = capture_image()
    status = detect_occupancy(image_path)
    print("Elevator Status:", status)

if __name__ == "__main__":
    main()
