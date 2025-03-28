import cv2
import numpy as np
import pyautogui
pyautogui.alert("Q-quit")
def track_green_area():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Camera not accessible")
        return
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Could not capture")
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        total_pixels = green_mask.shape[0] * green_mask.shape[1]
        green_pixels = cv2.countNonZero(green_mask)
        green_coverage = (green_pixels / total_pixels) * 100
        print(f" Green Space Coverage: {green_coverage:.2f}%")
        cv2.imshow('Live Feed', frame)
        cv2.imshow('Masked Feed', green_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting Green Area Tracker")
            break

    capture.release()
    cv2.destroyAllWindows()
track_green_area()
