import cv2
import numpy as np
import os

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture image.")
        return None

    if not os.path.exists("sample_images"):
        os.makedirs("sample_images")

    img_path = "sample_images/elevator.jpg"
    cv2.imwrite(img_path, frame)
    
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
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

    result = "FULL" if green_ratio < 0.2 else "NOT FULL"

    cv2.imshow("Processed Image", mask)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    return result

def main():
    image_path = capture_image()
    if image_path:
        status = detect_occupancy(image_path)
        print("Elevator Status:", status)

if __name__ == "__main__":
    main()
