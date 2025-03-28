
import cv2
import numpy as np
import time
import os
import json
import logging
from datetime import datetime

class ElevatorMonitor:
    def __init__(self, config_path="elevator_config.json"):
        self.start_logging()
        self.log = logging.getLogger("ElevatorMonitor")
        self.log.info("Starting elevator monitor")
        self.config_file = config_path
        self.load_settings()
        self.setup_cam()
        self.status = "Starting"
        self.old_status = None
        self.stable_count = 0
        self.last_change = time.time()
        self.active = True
        self.skip = 2
        self.frame_num = 0
        self.last_time = time.time()
        self.fps = 0
        self.persp_matrix = None
        self.out_w = 400
        self.out_h = 400
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=self.config['bg_history'],varThreshold=self.config['bg_threshold'],detectShadows=False)
        self.calib_mode = "notdone" if not self.config['is_calibrated'] else "running"
        self.mat_shape = None
        self.mat_size = 0
        self.no_gui = self.config.get('headless', False)
        self.last_save = time.time()
        self.save_every = 60

    def start_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        logname = f"logs/elevator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler(logname),logging.StreamHandler()])

    def load_settings(self):
        defaults = {'cam_idx': 0,'cam_w': 640,'cam_h': 480,'cam_fps': 15,'low_green': [40, 40, 40],'high_green': [85, 255, 255],'min_carpet': 0.30,'max_carpet': 0.80,'stable_limit': 15,'bg_hist': 500,'bg_thresh': 16,'adapt_thresh': True,'is_calibrated': False,'persp_pts': None,'roi': None,'mat_area': 0,'headless': False}
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    defaults.update(loaded)
                    self.log.info("Settings loaded")
            else:
                self.log.warning(f"No config at {self.config_file}. Using defaults")
        except Exception as e:
            self.log.error(f"Settings load failed: {str(e)}. Going with defaults")
        self.config = defaults
        self.low_green = np.array(self.config['low_green'])
        self.high_green = np.array(self.config['high_green'])
        self.min_carpet = self.config['min_carpet']
        self.max_carpet = self.config['max_carpet']
        self.stable_limit = self.config['stable_limit']

    def save_settings(self):
        try:
            self.config['low_green'] = self.low_green.tolist()
            self.config['high_green'] = self.high_green.tolist()
            self.config['min_carpet'] = self.min_carpet
            self.config['max_carpet'] = self.max_carpet
            self.config['stable_limit'] = self.stable_limit
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.log.info("Settings saved")
        except Exception as e:
            self.log.error(f"Save failed: {str(e)}")

    def setup_cam(self):
        tries = 0
        max_tries = 3
        while tries < max_tries:
            try:
                self.cap = cv2.VideoCapture(self.config['cam_idx'])
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['cam_w'])
                self.cap.set(4, self.config['cam_h'])
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                if not self.cap.isOpened():
                    raise Exception("Cam not working")
                ok, frame = self.cap.read()
                if ok:
                    h, w = frame.shape[:2]
                    self.w = w
                    self.h = h
                    print(f"Cam ready: {w}x{h}")
                    return
                else:
                    raise Exception("No frame from cam")
            except Exception as e:
                tries += 1
                self.log.warning(f"Cam try {tries} failed: {str(e)}")
                time.sleep(2)
        self.log.error("Cam setup failed after tries")
        if not self.no_gui:
            print("Cam not working. Check it.")
        self.w = 640
        self.h = 480

    def do_perspective(self, frame):
        if self.persp_matrix is not None:
            return cv2.warpPerspective(frame,self.persp_matrix,(self.out_w, self.out_h))
        return frame

    def set_perspective(self, pts):
        dst = np.array([[0, 0],[self.out_w - 1, 0],[self.out_w - 1, self.out_h - 1],[0, self.out_h - 1]], dtype=np.float32)
        self.persp_matrix = cv2.getPerspectiveTransform(pts, dst)
        self.config['persp_pts'] = pts.tolist()
        self.log.info("Perspective set up")

    def adjust_green_low(self, bright, low):
        if bright < 100:
            low[1] = max(20, low[1] - 20)
        return low

    def adjust_green_high(self, bright, high):
        if bright > 200:
            high[2] = min(100, high[2] + 20)
        return high

    def get_green_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.config['adapt_thresh']:
            bright = np.mean(hsv[:, :, 2])
            low = self.low_green.copy()
            high = self.high_green.copy()
            low = self.adjust_green_low(bright, low)
            high = self.adjust_green_high(bright, high)
            mask = cv2.inRange(hsv, low, high)
        else:
            mask = cv2.inRange(hsv, self.low_green, self.high_green)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def find_mat(self, frame):
        mask = self.get_green_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, mask
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            if cv2.contourArea(c) < 0.01 * self.w * self.h:
                continue
            eps = 0.05 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            if len(approx) >= 4 and len(approx) <= 6 and cv2.isContourConvex(approx):
                if len(approx) > 4:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    approx = box
                x, y, w, h = cv2.boundingRect(approx)
                ratio = w / h
                if 0.2 <= ratio <= 5.0:
                    return approx, (x, y, w, h), mask
        return None, None, mask

    def calibrate(self, frame):
        if self.no_gui:
            self.log.warning("Cant calibrate")
            return False
        shape, roi, mask = self.find_mat(frame)
        if shape is None:
            cv2.putText(frame, "carpet not found",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False
        cv2.drawContours(frame, [shape], -1, (0, 255, 0), 2)
        points = self.sort_points(shape)
        pts = np.float32(points)
        self.set_perspective(pts)
        new_mask = cv2.warpPerspective(mask,self.persp_matrix,(self.out_w, self.out_h))
        self.mat_size = cv2.countNonZero(new_mask)
        self.config['mat_area'] = self.mat_size
        self.config['roi'] = roi
        self.config['is_calibrated'] = True
        self.save_settings()
        cv2.putText(frame, "Calibrated. click -> 'c'.",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.log.info(f"Calibrated. Mat size: {self.mat_size}")
        return True

    def get_center_x(self, contour):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return 0
        return int(M['m10'] / M['m00'])

    def get_center_y(self, contour):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return 0
        return int(M['m01'] / M['m00'])

    def sort_points(self, contour):
        cx = self.get_center_x(contour)
        cy = self.get_center_y(contour)
        tl = None
        tr = None
        br = None
        bl = None
        min_d = float('inf')
        for pt in contour.reshape(-1, 2):
            x, y = pt
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if x < cx and y < cy:
                if tl is None or d < min_d:
                    tl = pt
                    min_d = d
            elif x > cx and y < cy:
                if tr is None or d < min_d:
                    tr = pt
                    min_d = d
            elif x > cx and y > cy:
                if br is None or d < min_d:
                    br = pt
                    min_d = d
            elif x < cx and y > cy:
                if bl is None or d < min_d:
                    bl = pt
                    min_d = d
        if tl is None:
            tl = [0, 0]
        if tr is None:
            tr = [self.w, 0]
        if br is None:
            br = [self.w, self.h]
        if bl is None:
            bl = [0, self.h]
        return np.array([tl, tr, br, bl])

    def check_status(self, ratio):
        if ratio < self.min_carpet:
            return "Crowded"
        elif ratio > self.max_carpet:
            return "Empty"
        else:
            return "Optimal"

    def update_status(self, new_stat):
        if new_stat == self.status:
            self.stable_count += 1
        else:
            self.stable_count = 0
        if self.stable_count >= self.stable_limit:
            if new_stat != self.status:
                self.old_status = self.status
                self.status = new_stat
                self.last_change = time.time()
                self.log.info(f"Status now {self.status} from {self.old_status}")
                if not self.no_gui:
                    if new_stat == "Crowded":
                        print(f"{datetime.now()}: the cart is crowded")
                    elif new_stat == "Empty":
                        print(f"{datetime.now()}: Cart is empty")
                    elif new_stat == "Optimal":
                        print(f"{datetime.now()}: optimal")

    def draw_roi(self, frame, roi):
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def process_frame(self, frame):
        if self.persp_matrix is not None:
            warped = self.do_perspective(frame)
            mask = self.get_green_mask(warped)
            green = cv2.countNonZero(mask)
            visibility = green / self.mat_size if self.mat_size > 0 else 0
            if not self.no_gui and self.config['roi'] is not None:
                self.draw_roi(frame, self.config['roi'])
            if not self.no_gui:
                cv2.imshow('Warped', warped)
                cv2.imshow('Mask', mask)
        else:
            if self.config['roi'] is None:
                if not self.no_gui:
                    cv2.putText(frame, "Not set up. Press 'c' to fix.",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.log.warning("Running without setup")
                return frame, 0
            x, y, w, h = self.config['roi']
            mask = self.get_green_mask(frame)
            roi_mask = mask[y:y + h, x:x + w] if y + h <= mask.shape[0] and x + w <= mask.shape[1] else None
            if roi_mask is None:
                self.log.error("ROI out of bounds")
                return frame, 0
            green = cv2.countNonZero(roi_mask)
            visibility = green / self.config['mat_area'] if self.config['mat_area'] > 0 else 0
            if not self.no_gui:
                self.draw_roi(frame, self.config['roi'])
        new_stat = self.check_status(visibility)
        self.update_status(new_stat)
        if not self.no_gui:
            cv2.putText(frame, f"Carpet seen: {visibility:.2%}",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            color = (0, 255, 0) if new_stat == "Good" else (0, 0, 255)
            cv2.putText(frame, f"Status: {new_stat}",(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}",(10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame, visibility

    def save_and_check(self):
        now = time.time()
        if now - self.last_save > self.save_every:
            self.save_settings()
            self.log.info(f"Status: {self.status}, steady for {now - self.last_change:.1f}s")
            self.last_save = now

    def get_fps(self):
        now = time.time()
        diff = now - self.last_time
        if diff > 0:
            self.fps = 1.0 / diff
        self.last_time = now

    def check_cam_status(self):
        if not self.cap.isOpened():
            self.log.warning("Reconnecting...")
            self.setup_cam()

    def handle_key(self, key):
        if key == ord('q'):
            self.active = False
        elif key == ord('c'):
            self.calib_mode = "fixing"
        elif key == ord('r'):
            self.calib_mode = "not done"
            self.log.info("Reset requested")

    def run(self):
        if not hasattr(self, 'cap') or (hasattr(self, 'cap') and not self.cap.isOpened()):
            self.log.error("No cam. Trying again...")
            self.setup_cam()
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                self.log.error("Cam still down. Stopping.")
                return
        if self.config['persp_pts'] is not None:
            pts = np.array(self.config['persp_pts'], dtype=np.float32)
            self.set_perspective(pts)
        self.log.info("Monitor starting")
        last_check = time.time()
        try:
            while self.active:
                if time.time() - last_check > 30:
                    self.check_cam_status()
                    last_check = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    self.log.warning("No frame. Cam might be off.")
                    time.sleep(1)
                    continue
                self.frame_num += 1
                if self.frame_num % (self.skip + 1) != 0 and self.calib_mode == "running":
                    if not self.no_gui and hasattr(self, 'last_frame'):
                        cv2.imshow('Elevator', self.last_frame)
                        key = cv2.waitKey(1) & 0xFF
                        self.handle_key(key)
                    continue
                self.get_fps()
                if self.calib_mode == "notdone" or self.calib_mode == "fixing":
                    done = self.calibrate(frame)
                    if done:
                        self.calib_mode = "running"
                        self.log.info("Now running after calibration")
                    if not self.no_gui:
                        cv2.imshow('Elevator', frame)
                    if not self.no_gui:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.active = False
                        elif key == ord('c') and done:
                            self.calib_mode = "running"
                elif self.calib_mode == "running":
                    new_frame, visibility = self.process_frame(frame)
                    self.last_frame = new_frame.copy()
                    self.save_and_check()
                    if not self.no_gui:
                        cv2.imshow('Elevator', new_frame)
                    if not self.no_gui:
                        key = cv2.waitKey(1) & 0xFF
                        self.handle_key(key)
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.log.info("Stopped by user")
        except Exception as e:
            self.log.error(f"Run error: {str(e)}")
            import traceback
            self.log.error(traceback.format_exc())
        finally:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            self.save_settings()
            self.log.info("Monitor shut down")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Elevator Occupancy Monitor')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    parser.add_argument('--config', type=str, default='elevator_config.json', help='Config file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera number')
    args = parser.parse_args()
    monitor = ElevatorMonitor(config_path=args.config)
    monitor.config['headless'] = args.headless
    monitor.config['cam_idx'] = args.camera
    monitor.no_gui = args.headless
    monitor.run()