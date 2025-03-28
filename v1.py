import cv2
import numpy as np
import os
import time
from datetime import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

THRESHOLDS = {
    10: "1O0YSXr2NKUKxO1__8ciSvMLQEy4cj66n",
    20: "1Ep_l1m5yL1v1ktrPq5qPTEDqDskAJK8m",
    30: "1gD6vm27exgug9ZHjDWAU9hjJQHKe6E-V",
    40: "1EOkiIWmBv3kM2ur5zxzWZeokmRuYo_hk",
    50: "1_Hnr5kkvj14bGCXiZis0LvHGt1gboaPm",
    60: "1R3qFOe3kQGMYOvVcMlErn3Vdmxf6R5yW",
    70: "1_fV8PuYD2x4EWXdSNfrYJrdvpw3RHhmN",
    80: "1x3UBwOFgl0fauCHBtgfa3-gXunQtqxn3",
    90: "1-O8hns9AtK19XNAOI58uiXMuBzVG9zKu",
    100: "1B3per9rnCyxvMHKTOJODp95PnWhvrvbm",
}

CONFIG = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "save_interval": 10,
    "min_area_for_calibration": 5000,
    "warped_width": 400,
    "warped_height": 300,
    "service_account_file": "C:/Users/levak/PycharmProjects/TeamCodeX/.venv/great.json",
    "fps_display_interval": 1.0,
    "upload_batch_size": 5,
    "enable_threading": True
}

class CarpetMonitor:
    def __init__(self):
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.camera = cv2.VideoCapture(CONFIG["camera_index"])
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.camera.isOpened():
            raise Exception("Error: Camera could not be opened!")
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([85, 255, 255])
        self.current_threshold = None
        self.perspective_matrix = None
        self.is_calibrated = False
        self.running = True
        self.last_save_time = 0
        self.last_uploaded_image_id = None
        self.warped_size = (CONFIG["warped_width"], CONFIG["warped_height"])
        self.total_pixels = CONFIG["warped_width"] * CONFIG["warped_height"]
        if CONFIG["enable_threading"]:
            self.setup_threading()
        self.drive = None
        self.setup_google_drive()
        self.result_frame = np.zeros((CONFIG["warped_height"], CONFIG["warped_width"], 3), dtype=np.uint8)
        self.overlay_frame = np.zeros((CONFIG["warped_height"], CONFIG["warped_width"], 3), dtype=np.uint8)
        self.image_queue = []

    def setup_threading(self):
        import threading
        self.upload_thread = None
        self.upload_lock = threading.Lock()

    def setup_google_drive(self):
        try:
            service_account_file = CONFIG["service_account_file"]
            print(f"Looking for service account file at: {service_account_file}")
            if not os.path.exists(service_account_file):
                print(f"Warning: Service account file not found at {service_account_file}")
                return
            scope = ['https://www.googleapis.com/auth/drive']
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                service_account_file, scope)
            gauth = GoogleAuth()
            gauth.credentials = credentials
            self.drive = GoogleDrive(gauth)
            print("Google Drive authentication successful with service account!")
        except Exception as e:
            print(f"Google Drive authentication failed: {e}")
            import traceback
            traceback.print_exc()
            self.drive = None

    def auto_calibrate(self):
        print("Attempting auto-calibration...")
        for _ in range(5):
            self.camera.read()
        start_time = time.time()
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts and time.time() - start_time < 5:
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > CONFIG["min_area_for_calibration"]:
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    self.calibrate_with_points(frame, box)
                    print(f"Auto-calibration successful after {attempt + 1} attempts!")
                    return True
            attempt += 1
            cv2.waitKey(100)
        print("Auto-calibration failed. Please calibrate manually by pressing 'c'.")
        return False

    def calibrate_frame(self, frame):
        print("Manual calibration started. Click on the four corners of the carpet area (clockwise from top-left).")
        points = []
        frame_copy = frame.copy()
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(frame_copy, tuple(points[i]), tuple(points[len(points) - 1]), (0, 255, 0), 2)
                    if len(points) == 4:
                        cv2.line(frame_copy, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
                cv2.imshow("Calibration", frame_copy)
                if len(points) == 4:
                    cv2.waitKey(500)
                    cv2.destroyWindow("Calibration")
                    self.calibrate_with_points(frame, np.array(points))
        cv2.namedWindow("Calibration")
        cv2.imshow("Calibration", frame_copy)
        cv2.setMouseCallback("Calibration", mouse_callback)
        while len(points) < 4 and self.running:
            key = cv2.waitKey(100) & 0xFF
            if key == 27:
                break
        cv2.destroyWindow("Calibration")

    def calibrate_with_points(self, frame, points):
        try:
            points = self.sort_points(points)
            height, width = CONFIG["warped_height"], CONFIG["warped_width"]
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            self.perspective_matrix = cv2.getPerspectiveTransform(
                np.array(points, dtype="float32"),
                dst_pts
            )
            self.is_calibrated = True
            print("Calibration complete!")
            np.save("calibration_points.npy", points)
            print("Calibration points saved for future use")
        except Exception as e:
            print(f"Calibration failed: {e}")

    def load_calibration(self):
        try:
            if os.path.exists("calibration_points.npy"):
                points = np.load("calibration_points.npy")
                _, frame = self.camera.read()
                if frame is not None:
                    self.calibrate_with_points(frame, points)
                    print("Successfully loaded calibration from file")
                    return True
                else:
                    print("Could not read frame from camera to apply calibration")
            else:
                print("No calibration file found")
        except Exception as e:
            print(f"Failed to load calibration: {e}")
        return False

    def sort_points(self, pts):
        pts = np.array(pts)
        rect = np.zeros((4, 2), dtype="float32")
        center_x = np.mean(pts[:, 0])
        center_y = np.mean(pts[:, 1])
        for i, pt in enumerate(pts):
            if pt[0] < center_x and pt[1] < center_y:
                rect[0] = pt
            elif pt[0] > center_x and pt[1] < center_y:
                rect[1] = pt
            elif pt[0] > center_x and pt[1] > center_y:
                rect[2] = pt
            else:
                rect[3] = pt
        return rect

    def get_threshold_bucket(self, percentage):
        for threshold in sorted(THRESHOLDS.keys()):
            if percentage <= threshold + 5:
                return threshold
        return 100

    def process_frame(self, frame):
        if not self.is_calibrated or self.perspective_matrix is None:
            return None
        warped = cv2.warpPerspective(frame, self.perspective_matrix, self.warped_size)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_pixels = cv2.countNonZero(mask)
        percentage = (green_pixels / self.total_pixels) * 100
        threshold = self.get_threshold_bucket(percentage)
        color_mask = np.zeros_like(warped)
        color_mask[:, :] = (0, 255, 0)
        green_overlay = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        result = cv2.addWeighted(warped, 0.7, green_overlay, 0.3, 0)
        cv2.putText(
            result,
            f"Green: {percentage:.1f}% (T: {threshold}%)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            result,
            f"FPS: {self.fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.imshow("Processed", result)
        return threshold, warped, mask, percentage

    def update_fps(self):
        current_time = time.time()
        self.frame_count += 1
        time_elapsed = current_time - self.fps_start_time
        if time_elapsed > CONFIG["fps_display_interval"]:
            self.fps = self.frame_count / time_elapsed
            self.frame_count = 0
            self.fps_start_time = current_time

    def save_image_async(self, exact_percentage, frame, threshold):
        if CONFIG["enable_threading"]:
            self.image_queue.append((exact_percentage, frame.copy(), threshold))
            if self.upload_thread is None or not self.upload_thread.is_alive():
                import threading
                self.upload_thread = threading.Thread(target=self.process_image_queue)
                self.upload_thread.daemon = True
                self.upload_thread.start()
        else:
            self.save_image(exact_percentage, frame)

    def process_image_queue(self):
        with self.upload_lock:
            batch = self.image_queue[:CONFIG["upload_batch_size"]]
            self.image_queue = self.image_queue[CONFIG["upload_batch_size"]:]
        for exact_percentage, frame, threshold in batch:
            self.save_image(exact_percentage, frame)
        if self.image_queue and self.running:
            import threading
            self.upload_thread = threading.Thread(target=self.process_image_queue)
            self.upload_thread.daemon = True
            self.upload_thread.start()

    def save_image(self, exact_percentage, frame):
        if self.drive is None:
            print("Google Drive is not connected. Cannot save image.")
            return None, None
        try:
            current_time = datetime.now()
            formatted_time = current_time.strftime("%H:%M:%S")
            formatted_date = current_time.strftime("%d-%m-%Y")
            percentage_str = f"{int(exact_percentage)}%"
            filename = f"{percentage_str}-{formatted_time}-{formatted_date}.jpg"
            filename = filename.replace(":", "-")
            threshold = self.get_threshold_bucket(exact_percentage)
            temp_dir = "temp_images"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_path = os.path.join(temp_dir, filename)
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            success = cv2.imwrite(temp_path, frame, encode_params)
            if not success:
                print("Failed to save temporary image")
                return None, None
            folder_id = THRESHOLDS[threshold]
            file_metadata = {
                'title': filename,
                'parents': [{'id': folder_id}]
            }
            file_drive = self.drive.CreateFile(file_metadata)
            file_drive.SetContentFile(temp_path)
            file_drive.Upload()
            print(f"Image uploaded: {filename} in {threshold}% folder")
            self.last_uploaded_image_id = file_drive['id']
            os.remove(temp_path)
            return threshold, file_drive['id']
        except Exception as e:
            print(f"Error uploading image to Google Drive: {e}")
            return None, None

    def adjust_green_thresholds(self):
        print("Adjusting green detection thresholds. Press 'q' when done.")
        cv2.namedWindow("HSV Adjustment")
        cv2.createTrackbar("L-H", "HSV Adjustment", self.lower_green[0], 179, lambda x: None)
        cv2.createTrackbar("L-S", "HSV Adjustment", self.lower_green[1], 255, lambda x: None)
        cv2.createTrackbar("L-V", "HSV Adjustment", self.lower_green[2], 255, lambda x: None)
        cv2.createTrackbar("U-H", "HSV Adjustment", self.upper_green[0], 179, lambda x: None)
        cv2.createTrackbar("U-S", "HSV Adjustment", self.upper_green[1], 255, lambda x: None)
        cv2.createTrackbar("U-V", "HSV Adjustment", self.upper_green[2], 255, lambda x: None)
        frame_delay = 1 / 15
        last_process_time = 0
        while True:
            current_time = time.time()
            if current_time - last_process_time < frame_delay:
                time.sleep(0.005)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last_process_time = current_time
            ret, frame = self.camera.read()
            if not ret:
                continue
            if self.is_calibrated:
                display_frame = cv2.warpPerspective(
                    frame,
                    self.perspective_matrix,
                    (CONFIG["warped_width"], CONFIG["warped_height"])
                )
            else:
                display_frame = frame
            l_h = cv2.getTrackbarPos("L-H", "HSV Adjustment")
            l_s = cv2.getTrackbarPos("L-S", "HSV Adjustment")
            l_v = cv2.getTrackbarPos("L-V", "HSV Adjustment")
            u_h = cv2.getTrackbarPos("U-H", "HSV Adjustment")
            u_s = cv2.getTrackbarPos("U-S", "HSV Adjustment")
            u_v = cv2.getTrackbarPos("U-V", "HSV Adjustment")
            lower_green_temp = np.array([l_h, l_s, l_v])
            upper_green_temp = np.array([u_h, u_s, u_v])
            hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green_temp, upper_green_temp)
            result = cv2.bitwise_and(display_frame, display_frame, mask=mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            green_pixels = cv2.countNonZero(mask)
            percentage = (green_pixels / total_pixels) * 100
            cv2.putText(
                result,
                f"Green: {percentage:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.imshow("Original", display_frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        self.lower_green = lower_green_temp
        self.upper_green = upper_green_temp
        print(f"New HSV thresholds: Lower {self.lower_green}, Upper {self.upper_green}")
        cv2.destroyWindow("HSV Adjustment")
        cv2.destroyWindow("Original")
        cv2.destroyWindow("Mask")
        cv2.destroyWindow("Result")

    def run(self):
        print("Starting carpet monitor...")
        print("Commands:")
        print("  'c' - Manual calibration")
        print("  'a' - Adjust green color detection")
        print("  'l' - Load saved calibration")
        print("  's' - Save current image")
        print("  'q' - Quit")
        if not self.load_calibration():
            self.auto_calibrate()
        last_frame_time = time.time()
        frame_delay = 1 / 30
        while self.running:
            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time
            if time_since_last_frame < frame_delay:
                sleep_time = max(0, frame_delay - time_since_last_frame)
                time.sleep(sleep_time * 0.8)
            last_frame_time = time.time()
            grabbed = self.camera.grab()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                ret, frame = self.camera.retrieve()
                if ret:
                    self.calibrate_frame(frame.copy())
            elif key == ord('a'):
                self.adjust_green_thresholds()
            elif key == ord('l'):
                self.load_calibration()
            elif key == ord('s'):
                if self.is_calibrated:
                    ret, frame = self.camera.retrieve()
                    if ret:
                        result = self.process_frame(frame)
                        if result:
                            _, _, _, exact_percentage = result
                            self.save_image(exact_percentage, frame)
            if not grabbed:
                print("Failed to grab frame from camera.")
                time.sleep(0.01)
                continue
            ret, frame = self.camera.retrieve()
            if not ret:
                continue
            self.update_fps()
            if self.is_calibrated:
                result = self.process_frame(frame)
                if result:
                    threshold, warped, mask, exact_percentage = result
                    current_time = time.time()
                    time_since_last_save = current_time - self.last_save_time
                    if (threshold != self.current_threshold or
                            time_since_last_save > CONFIG["save_interval"]):
                        if threshold != self.current_threshold:
                            print(f"Threshold changed from {self.current_threshold} to {threshold}")
                        self.save_image_async(exact_percentage, frame, threshold)
                        self.current_threshold = threshold
                        self.last_save_time = current_time
            else:
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    "NOT CALIBRATED - Press 'c' to calibrate",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    display_frame,
                    f"FPS: {self.fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.imshow("Original", display_frame)
        if CONFIG["enable_threading"] and self.upload_thread and self.upload_thread.is_alive():
            print("Waiting for upload thread to finish...")
            self.upload_thread.join(timeout=3.0)
        self.camera.release()
        cv2.destroyAllWindows()
        print("Carpet monitor stopped")

if __name__ == "__main__":
    monitor = CarpetMonitor()
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()