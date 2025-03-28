import cv2
import numpy as np
import time
import pyautogui
pyautogui.alert("press Q to quit")
def auto_calibrate_green(cap):
    print("ACTIVE")
    lower_bounds = []
    upper_bounds = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        if cv2.countNonZero(mask) > 0:
            mean_hsv = cv2.mean(hsv, mask=mask)[:3]
            lower_bounds.append([max(35, mean_hsv[0] - 10), 40, 40])
            upper_bounds.append([min(85, mean_hsv[0] + 10), 255, 255])
    if lower_bounds and upper_bounds:
        lower_green = np.median(np.array(lower_bounds), axis=0).astype(int)
        upper_green = np.median(np.array(upper_bounds), axis=0).astype(int)
    else:
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
    return lower_green, upper_green
def track_green_area():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("unable to access camera")
        return
    lower_green, upper_green = auto_calibrate_green(cap)
    last_print_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("unable to capture frames")
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        total_pixels = green_mask.shape[0] * green_mask.shape[1]
        green_pixels = cv2.countNonZero(green_mask)
        green_coverage = (green_pixels / total_pixels) * 100
        current_time = time.time()
        if current_time - last_print_time >= 3:
            last_print_time = current_time
        result_frame = frame.copy()
        cv2.putText(result_frame, f"Green: {green_coverage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        cv2.imshow('live Feed', result_frame)
        cv2.imshow('masked view', green_mask)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break
    cap.release()
    cv2.destroyAllWindows()
track_green_area()
