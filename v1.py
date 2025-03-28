import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
import logging
class GreenTracker:
    def __init__(self, config_path="tracker_config.json"):
        self.setup_logging()
        self.logger = logging.getLogger("GreenTracker")
        self.logger.info("Initializing Green Tracker")
        self.config_path = config_path
        self.load_configuration()
        self.setup_camera()
        self.lower_green = np.array(self.config['lower_green'])
        self.upper_green = np.array(self.config['upper_green'])
        self.last_print_time = time.time()
        self.fps = 0
        self.last_processed_time = time.time()
        self.running = True
        self.skip_frames = 2
        self.frame_count = 0

    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_file = f"logs/tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )

    def load_configuration(self):
        default_config = {
            'camera_index': 0,
            'camera_width': 640,
            'camera_height': 480,
            'camera_fps': 15,
            'lower_green': [35, 40, 40],
            'upper_green': [85, 255, 255],
            'min_area': 500
        }
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = default_config
                self.save_configuration()
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            self.config = default_config

    def save_configuration(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")

    def setup_camera(self):
        self.cap = cv2.VideoCapture(self.config['camera_index'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['camera_fps'])
        if not self.cap.isOpened():
            self.logger.error("Camera failed to initialize")
            raise Exception("Camera not available")

    def auto_calibrate_green(self):
        self.logger.info("Starting auto-calibration")
        lower_bounds = []
        upper_bounds = []
        for _ in range(30):
            ret, frame = self.cap.read()
            if not ret:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            if cv2.countNonZero(mask) > 0:
                mean_hsv = cv2.mean(hsv, mask=mask)[:3]
                lower_bounds.append([max(35, mean_hsv[0] - 10), 40, 40])
                upper_bounds.append([min(85, mean_hsv[0] + 10), 255, 255])
        if lower_bounds and upper_bounds:
            self.lower_green = np.median(np.array(lower_bounds), axis=0).astype(int)
            self.upper_green = np.median(np.array(upper_bounds), axis=0).astype(int)
            self.config['lower_green'] = self.lower_green.tolist()
            self.config['upper_green'] = self.upper_green.tolist()
            self.save_configuration()
        else:
            self.lower_green = np.array([35, 40, 40])
            self.upper_green = np.array([85, 255, 255])
        self.logger.info("Calibration complete")

    def get_adaptive_green_mask(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_brightness = np.mean(hsv_frame[:, :, 2])
        adjusted_lower = self.lower_green.copy()
        adjusted_upper = self.upper_green.copy()
        if frame_brightness < 100:
            adjusted_lower[1] = max(20, adjusted_lower[1] - 20)
        if frame_brightness > 200:
            adjusted_lower[2] = min(100, adjusted_lower[2] + 20)
        mask = cv2.inRange(hsv_frame, adjusted_lower, adjusted_upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def calculate_fps(self):
        current_time = time.time()
        time_diff = current_time - self.last_processed_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.last_processed_time = current_time

    def process_frame(self, frame):
        green_mask = self.get_adaptive_green_mask(frame)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.config['min_area']:
                x, y, w, h = cv2.boundingRect(largest_contour)
                frame = frame[y:y + h, x:x + w]
                green_mask = green_mask[y:y + h, x:x + w]
        total_pixels = frame.shape[0] * frame.shape[1]
        green_pixels = cv2.countNonZero(green_mask)
        green_coverage = (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        return frame, green_mask, green_coverage

    def run(self):
        self.logger.info("Starting tracking loop")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to grab frame")
                time.sleep(1)
                continue

            self.frame_count += 1
            if self.frame_count % (self.skip_frames + 1) != 0:
                cv2.imshow('Live Camera Feed', self.last_frame) if hasattr(self, 'last_frame') else None
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q')]:
                    break
                continue

            self.calculate_fps()
            result_frame, green_mask, green_coverage = self.process_frame(frame)
            self.last_frame = result_frame.copy()

            current_time = time.time()
            if current_time - self.last_print_time >= 3:
                self.logger.info(f"Green coverage: {green_coverage:.2f}%")
                self.last_print_time = current_time

            cv2.putText(result_frame, f"Green: {green_coverage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"FPS: {self.fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Live Camera Feed', result_frame)
            cv2.imshow('Green Detection Mask', green_mask)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                self.running = False
            elif key in [ord('c'), ord('C')]:
                self.auto_calibrate_green()

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_configuration()
        self.logger.info("Tracker shutdown complete")

if __name__ == "__main__":
    tracker = GreenTracker()
    tracker.run()