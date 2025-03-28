import cv2
import numpy as np
import os
import time
from datetime import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import OAuth2WebServerFlow
import pickle

# Google Drive folder IDs
GREAT_FOLDER_ID = None  # Main GREAT folder ID
THRESHOLD_FOLDER_IDS = {}  # Will store subfolder IDs for each threshold (10%, 20%, etc.)
THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Threshold percentages


# Configuration
CONFIG = {
"camera_index": 0,  # Change if using a different camera
"camera_width": 640,
"camera_height": 480,
"save_interval": 10,  # Seconds between automatic saves
"min_area_for_calibration": 5000,  # Minimum green area for auto-calibration
"warped_width": 400,  # Width of perspective-transformed image
"warped_height": 300,  # Height of perspective-transformed image
"fps_display_interval": 1.0,  # How often to update FPS display (seconds)
"upload_batch_size": 5,  # Number of images to batch process before uploading
"enable_threading": True,  # Use threading for non-critical operations
"target_fps": 20,  # Target FPS for processing
"video_segment_duration": 120,  # Duration of each video segment in seconds (2 minutes)
"video_fps": 30,  # FPS for video recording
"video_codec": cv2.VideoWriter_fourcc(*'mp4v'),  # Video codec for MP4
"video_dir": os.path.join("saved_images", "GREAT", "video")  # Video directory path
}




class CarpetMonitor:
   def __init__(self):
       # Performance tracking
       self.frame_count = 0
       self.fps = 0
       self.fps_start_time = time.time()

       # Camera setup with optimized buffer size
       self.camera = None
       self.initialize_camera()

       # Green color detection thresholds (HSV)
       self.lower_green = np.array([40, 40, 40])
       self.upper_green = np.array([85, 255, 255])

       # Current threshold bucket and tracking
       self.current_threshold = None
       self.perspective_matrix = None
       self.is_calibrated = False
       self.running = True
       self.last_save_time = 0
       self.last_uploaded_image_id = None
       self.calibration_mode = False  # Added for CLI calibration

       # Image processing optimization
       self.warped_size = (CONFIG["warped_width"], CONFIG["warped_height"])
       self.total_pixels = CONFIG["warped_width"] * CONFIG["warped_height"]

       # Video recording setup
       self.video_writer = None
       self.video_start_time = None
       self.current_video_filename = None
       self.setup_video_directory()

       # Threading support
       if CONFIG["enable_threading"]:
           self.setup_threading()

       # Google Drive setup
       self.drive = None
       self.setup_google_drive()

       # Pre-allocate reusable matrices for optimization
       self.result_frame = np.zeros((CONFIG["warped_height"], CONFIG["warped_width"], 3), dtype=np.uint8)
       self.overlay_frame = np.zeros((CONFIG["warped_height"], CONFIG["warped_width"], 3), dtype=np.uint8)

       # Image queue for batch processing
       self.image_queue = []

   def initialize_camera(self):
       """Initialize the camera with proper settings"""
       try:
           self.camera = cv2.VideoCapture(CONFIG["camera_index"])
           if not self.camera.isOpened():
               raise Exception("Error: Camera could not be opened!")
           
           # Set camera properties
           self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
           self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
           self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
           
           # Verify camera settings
           actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
           actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
           if actual_width != CONFIG["camera_width"] or actual_height != CONFIG["camera_height"]:
               print(f"Warning: Camera resolution mismatch. Expected: {CONFIG['camera_width']}x{CONFIG['camera_height']}, Got: {actual_width}x{actual_height}")
           
           print("Camera initialized successfully")
       except Exception as e:
           print(f"Error initializing camera: {e}")
           if self.camera:
               self.camera.release()
           raise

   def __del__(self):
       """Cleanup when the object is destroyed"""
       try:
           self.stop_recording()
           if self.camera:
               self.camera.release()
           cv2.destroyAllWindows()
       except Exception as e:
           print(f"Error during cleanup: {e}")

   def setup_threading(self):
       """Setup threading for non-critical operations"""
       import threading
       self.upload_thread = None
       self.upload_lock = threading.Lock()


   def setup_google_drive(self):
       """Initialize Google Drive connection using OAuth for user account"""
       global GREAT_FOLDER_ID
       try:
           # Check for saved credentials
           credentials_file = 'google_drive_credentials.pickle'
           
           gauth = GoogleAuth()
           
           # Try to load saved client credentials
           if os.path.exists(credentials_file):
               with open(credentials_file, 'rb') as token:
                   credentials = pickle.load(token)
                   gauth.credentials = credentials
               print("Loaded saved Google Drive credentials")
           else:
               # Set up the OAuth flow
               # Note: This will open a browser window for authentication
               gauth.LocalWebserverAuth()  # Creates a local webserver and automatically handles authentication
               
               # Save the credentials for future use
               with open(credentials_file, 'wb') as token:
                   pickle.dump(gauth.credentials, token)
               print("Saved new Google Drive credentials")
           
           # Initialize the Google Drive instance
           self.drive = GoogleDrive(gauth)
           print("Google Drive authentication successful!")
           
           # Find or create the "GREAT" folder
           self._find_or_create_great_folder()
           
       except Exception as e:
           print(f"Google Drive authentication failed: {e}")
           import traceback
           traceback.print_exc()
           self.drive = None
           
           # Create local directory for saving images as fallback
           local_dir = "saved_images"
           if not os.path.exists(local_dir):
               os.makedirs(local_dir)
               os.makedirs(os.path.join(local_dir, "GREAT"), exist_ok=True)
           print(f"Created local directory for saving images: {local_dir}")
   
   def _find_or_create_great_folder(self):
       """Find or create the GREAT folder and threshold subfolders in Google Drive"""
       global GREAT_FOLDER_ID, THRESHOLD_FOLDER_IDS
       
       if self.drive is None:
           return
           
       try:
           # Search for the GREAT folder
           query = "title='GREAT' and mimeType='application/vnd.google-apps.folder' and trashed=false"
           file_list = self.drive.ListFile({'q': query}).GetList()
           
           if file_list:
               # Folder exists, use its ID
               GREAT_FOLDER_ID = file_list[0]['id']
               print(f"Found existing GREAT folder with ID: {GREAT_FOLDER_ID}")
           else:
               # Create the GREAT folder
               folder_metadata = {
                   'title': 'GREAT',
                   'mimeType': 'application/vnd.google-apps.folder'
               }
               folder = self.drive.CreateFile(folder_metadata)
               folder.Upload()
               GREAT_FOLDER_ID = folder['id']
               print(f"Created new GREAT folder with ID: {GREAT_FOLDER_ID}")
           
           # Now find or create threshold subfolders
           for threshold in THRESHOLDS:
               subfolder_name = f"{threshold}"
               # Search for the threshold subfolder
               query = f"title='{subfolder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false and '{GREAT_FOLDER_ID}' in parents"
               file_list = self.drive.ListFile({'q': query}).GetList()
               
               if file_list:
                   # Subfolder exists, use its ID
                   THRESHOLD_FOLDER_IDS[threshold] = file_list[0]['id']
                   print(f"Found existing {subfolder_name} subfolder with ID: {THRESHOLD_FOLDER_IDS[threshold]}")
               else:
                   # Create the threshold subfolder
                   subfolder_metadata = {
                       'title': subfolder_name,
                       'mimeType': 'application/vnd.google-apps.folder',
                       'parents': [{'id': GREAT_FOLDER_ID}]
                   }
                   subfolder = self.drive.CreateFile(subfolder_metadata)
                   subfolder.Upload()
                   THRESHOLD_FOLDER_IDS[threshold] = subfolder['id']
                   print(f"Created new {subfolder_name} subfolder with ID: {THRESHOLD_FOLDER_IDS[threshold]}")
               
       except Exception as e:
           print(f"Error finding/creating folders: {e}")
           import traceback
           traceback.print_exc()
           GREAT_FOLDER_ID = None
           THRESHOLD_FOLDER_IDS = {}

   def setup_video_directory(self):
       """Create video directory if it doesn't exist"""
       try:
           # Create local video directory for temporary storage
           if not os.path.exists(CONFIG["video_dir"]):
               os.makedirs(CONFIG["video_dir"])
               print(f"Created temporary video directory: {CONFIG['video_dir']}")
           else:
               print(f"Found existing temporary video directory: {CONFIG['video_dir']}")
               
           # Check if directory is writable
           test_file = os.path.join(CONFIG["video_dir"], "test.txt")
           with open(test_file, 'w') as f:
               f.write("test")
           os.remove(test_file)
           print("Video directory is writable")
           
       except Exception as e:
           print(f"Error setting up video directory: {e}")
           print("Video recording will be disabled")
           self.video_writer = None
           self.video_start_time = None
           self.current_video_filename = None

   def start_new_video_segment(self):
       """Start a new video segment"""
       try:
           # Close existing video writer if any
           if self.video_writer:
               self.video_writer.release()
               print(f"Closed previous video segment: {self.current_video_filename}")

           # Generate filename with date and time
           current_time = datetime.now()
           date_str = current_time.strftime("%Y-%m-%d")
           time_str = current_time.strftime("%H-%M-%S")
           self.current_video_filename = f"recording_{date_str}_{time_str}.mp4"
           
           # Create full path for video file
           video_path = os.path.join(CONFIG["video_dir"], self.current_video_filename)
           
           # Initialize video writer with original camera resolution
           self.video_writer = cv2.VideoWriter(
               video_path,
               CONFIG["video_codec"],
               CONFIG["video_fps"],
               (CONFIG["camera_width"], CONFIG["camera_height"])
           )
           
           if not self.video_writer.isOpened():
               raise Exception("Failed to create video writer")
           
           self.video_start_time = time.time()
           print(f"Started new video segment: {self.current_video_filename}")
           
       except Exception as e:
           print(f"Error starting new video segment: {e}")
           if self.video_writer:
               self.video_writer.release()
           self.video_writer = None
           self.video_start_time = None
           self.current_video_filename = None

   def upload_video_to_drive(self, video_path):
       """Upload video to Google Drive"""
       try:
           if self.drive is None or GREAT_FOLDER_ID is None:
               print("Google Drive not connected, skipping video upload")
               return

           # Create video folder if it doesn't exist
           video_folder_query = "title='video' and mimeType='application/vnd.google-apps.folder' and trashed=false and '" + GREAT_FOLDER_ID + "' in parents"
           video_folders = self.drive.ListFile({'q': video_folder_query}).GetList()
            
           if video_folders:
               video_folder_id = video_folders[0]['id']
           else:
               # Create video folder
               folder_metadata = {
                   'title': 'video',
                   'mimeType': 'application/vnd.google-apps.folder',
                   'parents': [{'id': GREAT_FOLDER_ID}]
               }
               video_folder = self.drive.CreateFile(folder_metadata)
               video_folder.Upload()
               video_folder_id = video_folder['id']

           # Upload the video file
           file_metadata = {
               'title': os.path.basename(video_path),
               'parents': [{'id': video_folder_id}]
           }
           file_drive = self.drive.CreateFile(file_metadata)
           file_drive.SetContentFile(video_path)
           file_drive.Upload()
           print(f"Video uploaded to Google Drive: {os.path.basename(video_path)}")

           # Remove local file after successful upload
           os.remove(video_path)
           print(f"Removed local video file: {video_path}")

       except Exception as e:
           print(f"Error uploading video to Google Drive: {e}")
           print(f"Video remains in local directory: {video_path}")

   def stop_recording(self):
       """Safely stop video recording and save the current segment"""
       try:
           if self.video_writer and self.video_writer.isOpened():
               self.video_writer.release()
               video_path = os.path.join(CONFIG["video_dir"], self.current_video_filename)
               print(f"Saved video segment: {self.current_video_filename}")
               
               # Upload to Google Drive
               self.upload_video_to_drive(video_path)
                
               self.video_writer = None
               self.video_start_time = None
               self.current_video_filename = None
       except Exception as e:
           print(f"Error stopping video recording: {e}")

   def check_video_segment(self):
       """Check if we need to start a new video segment"""
       if self.video_writer is None:
           self.start_new_video_segment()
       elif time.time() - self.video_start_time >= CONFIG["video_segment_duration"]:
           print("Video segment duration reached, starting new segment")
           self.start_new_video_segment()

   def record_frame(self, frame):
       """Record a frame to the current video segment"""
       try:
           if self.video_writer and self.video_writer.isOpened():
               # Add timestamp overlay to the frame
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               cv2.putText(
                   frame,
                   timestamp,
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (255, 255, 255),
                   2
               )
               self.video_writer.write(frame)
       except Exception as e:
           print(f"Error recording frame: {e}")

   def auto_calibrate(self):
       """Attempt to automatically calibrate the system by finding a green area"""
       print("Attempting auto-calibration...")


       # Clear buffer frames for fresh input
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


           # Convert to HSV and find green areas
           hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           mask = cv2.inRange(hsv, self.lower_green, self.upper_green)


           # Find contours in the mask
           contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


           if contours:
               # Find the largest contour
               largest_contour = max(contours, key=cv2.contourArea)
               area = cv2.contourArea(largest_contour)


               if area > CONFIG["min_area_for_calibration"]:  # Minimum area threshold
                   # Get bounding rectangle
                   rect = cv2.minAreaRect(largest_contour)
                   box = cv2.boxPoints(rect)
                   box = box.astype(np.int32)


                   # Use the bounding box for calibration
                   self.calibrate_with_points(frame, box)
                   print(f"Auto-calibration successful after {attempt + 1} attempts!")
                   return True


           attempt += 1
           cv2.waitKey(100)  # Small delay to allow camera to adjust


       print("Auto-calibration failed. Please calibrate manually by pressing 'c'.")
       return False


   def calibrate_frame(self, frame):
       """Manual calibration process"""
       print("Manual calibration started. Click on the four corners of the carpet area (clockwise from top-left).")
       points = []
       frame_copy = frame.copy()  # Create a copy to avoid modifying the original
       
       # Pre-draw instructions on the frame
       cv2.putText(
           frame_copy,
           "Click on 4 corners (clockwise from top-left)",
           (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7,
           (0, 255, 0),
           2
       )
       
       cv2.putText(
           frame_copy,
           "Press ESC to cancel",
           (10, 60),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7,
           (0, 255, 0),
           2
       )

       def mouse_callback(event, x, y, flags, param):
           nonlocal frame_copy
           if event == cv2.EVENT_LBUTTONDOWN:
               points.append([x, y])
               # Create a fresh copy to avoid drawing over previous drawings
               frame_copy = frame.copy()
               
               # Redraw instructions
               cv2.putText(
                   frame_copy,
                   "Click on 4 corners (clockwise from top-left)",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 255, 0),
                   2
               )
               
               cv2.putText(
                   frame_copy,
                   "Press ESC to cancel",
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 255, 0),
                   2
               )
               
               # Draw all points and lines
               for i, pt in enumerate(points):
                   cv2.circle(frame_copy, (pt[0], pt[1]), 5, (0, 255, 0), -1)
                   cv2.putText(
                       frame_copy,
                       str(i+1),
                       (pt[0] + 10, pt[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2
                   )
               
               # Draw lines between points
               for i in range(len(points) - 1):
                   cv2.line(frame_copy, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 2)
               
               # Connect the last point to the first if we have 4 points
               if len(points) == 4:
                   cv2.line(frame_copy, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
               
               cv2.imshow("Calibration", frame_copy)
               
               if len(points) == 4:
                   cv2.waitKey(500)  # Brief delay to show the completed quadrilateral
                   try:
                       self.calibrate_with_points(frame, np.array(points))
                       cv2.destroyWindow("Calibration")
                   except Exception as e:
                       print(f"Calibration failed: {e}")
                       # Reset points and continue
                       points.clear()
                       frame_copy = frame.copy()
                       cv2.putText(
                           frame_copy,
                           "Calibration failed, try again",
                           (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (0, 0, 255),
                           2
                       )
                       cv2.imshow("Calibration", frame_copy)

       try:
           cv2.namedWindow("Calibration")
           cv2.imshow("Calibration", frame_copy)
           cv2.setMouseCallback("Calibration", mouse_callback)
           
           while len(points) < 4 and self.running:
               key = cv2.waitKey(20) & 0xFF  # More responsive key checking
               if key == 27:  # ESC key to cancel
                   break
           
           cv2.destroyWindow("Calibration")
       except Exception as e:
           print(f"Error during calibration: {e}")
           try:
               cv2.destroyWindow("Calibration")
           except:
               pass


   def calibrate_with_points(self, frame, points):
       """Set up perspective transform with the given points"""
       try:
           # Validate points
           if points is None or len(points) != 4:
               print(f"Invalid calibration points: expected 4 points, got {len(points) if points is not None else 0}")
               return False
               
           # Sort points to ensure correct order: top-left, top-right, bottom-right, bottom-left
           sorted_points = self.sort_points(points)
           
           if sorted_points is None:
               print("Failed to sort calibration points")
               return False

           # Define destination points (rectangle)
           height, width = CONFIG["warped_height"], CONFIG["warped_width"]
           dst_pts = np.array([
               [0, 0],
               [width - 1, 0],
               [width - 1, height - 1],
               [0, height - 1]
           ], dtype="float32")

           # Calculate perspective transform matrix
           try:
               self.perspective_matrix = cv2.getPerspectiveTransform(
                   np.array(sorted_points, dtype="float32"),
                   dst_pts
               )
           except Exception as e:
               print(f"Failed to calculate perspective transform: {e}")
               return False

           self.is_calibrated = True
           print("Calibration complete!")

           # Save calibration points to file for future use
           try:
               np.save("calibration_points.npy", sorted_points)
               print("Calibration points saved for future use")
           except Exception as e:
               print(f"Failed to save calibration points: {e}")
               # Continue even if saving fails
               
           return True

       except Exception as e:
           print(f"Calibration failed: {e}")
           import traceback
           traceback.print_exc()
           return False


   def load_calibration(self):
       """Load calibration from saved file if available"""
       try:
           if os.path.exists("calibration_points.npy"):
               try:
                   points = np.load("calibration_points.npy")
                   
                   # Validate loaded points
                   if points is None or len(points) != 4:
                       print(f"Invalid calibration points in file: expected 4 points, got {len(points) if points is not None else 0}")
                       return False
                       
                   # Read a frame from camera
                   for _ in range(3):  # Try up to 3 times to get a valid frame
                       _, frame = self.camera.read()
                       if frame is not None:
                           break
                       time.sleep(0.1)
                       
                   if frame is not None:
                       # Apply calibration with the loaded points
                       success = self.calibrate_with_points(frame, points)
                       if success:
                           print("Successfully loaded calibration from file")
                           return True
                       else:
                           print("Failed to apply calibration with loaded points")
                   else:
                       print("Could not read frame from camera to apply calibration")
               except Exception as e:
                   print(f"Error loading calibration file: {e}")
                   import traceback
                   traceback.print_exc()
           else:
               print("No calibration file found")
       except Exception as e:
           print(f"Failed to load calibration: {e}")
           import traceback
           traceback.print_exc()
       return False


   def sort_points(self, pts):
       """Sort points in order: top-left, top-right, bottom-right, bottom-left"""
       try:
           # Validate input
           if pts is None or len(pts) != 4:
               print(f"Invalid points for sorting: expected 4 points, got {len(pts) if pts is not None else 0}")
               return None
               
           # Convert points to a numpy array if they're not already
           pts = np.array(pts)
           
           # Ensure points are in the correct shape
           if pts.shape[0] != 4 or pts.shape[1] != 2:
               print(f"Invalid points shape: {pts.shape}, expected (4, 2)")
               return None

           # Get center of points
           rect = np.zeros((4, 2), dtype="float32")
           center_x = np.mean(pts[:, 0])
           center_y = np.mean(pts[:, 1])

           # Track which quadrants have been assigned
           assigned = [False, False, False, False]

           # Classify points relative to center
           for i, pt in enumerate(pts):
               # top-left: x < center_x and y < center_y
               # top-right: x > center_x and y < center_y
               # bottom-right: x > center_x and y > center_y
               # bottom-left: x < center_x and y > center_y
               if pt[0] < center_x and pt[1] < center_y:
                   rect[0] = pt
                   assigned[0] = True
               elif pt[0] > center_x and pt[1] < center_y:
                   rect[1] = pt
                   assigned[1] = True
               elif pt[0] > center_x and pt[1] > center_y:
                   rect[2] = pt
                   assigned[2] = True
               else:
                   rect[3] = pt
                   assigned[3] = True

           # Check if all quadrants were assigned
           if not all(assigned):
               print("Warning: Not all quadrants were assigned a point. Calibration may be inaccurate.")
               # Try to recover by using the points in order
               unassigned = [i for i, a in enumerate(assigned) if not a]
               remaining_pts = [p for p in pts if not any(np.array_equal(p, rect[i]) for i in range(4) if assigned[i])]
               
               for i, pt in zip(unassigned, remaining_pts):
                   rect[i] = pt
                   
           return rect
           
       except Exception as e:
           print(f"Error sorting points: {e}")
           import traceback
           traceback.print_exc()
           return None


   def get_threshold_bucket(self, percentage):
       """Determine which threshold bucket a percentage falls into"""
       thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
       for threshold in sorted(thresholds):
           if percentage <= threshold + 5:
               return threshold
       return 100


   def process_frame(self, frame):
       """Process a frame to detect green percentage (optimized)"""
       if not self.is_calibrated or self.perspective_matrix is None:
           return None

       # Apply perspective transform
       warped = cv2.warpPerspective(frame, self.perspective_matrix, self.warped_size)

       # Convert to HSV and find green areas
       hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
       mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

       # Calculate percentage of green
       green_pixels = cv2.countNonZero(mask)
       percentage = (green_pixels / self.total_pixels) * 100

       # Determine threshold bucket
       threshold = self.get_threshold_bucket(percentage)

       # Create a visualization
       color_mask = np.zeros_like(warped)
       color_mask[:, :] = (0, 255, 0)  # Green color

       # Apply mask to the color overlay
       green_overlay = cv2.bitwise_and(color_mask, color_mask, mask=mask)

       # Combine original and overlay
       result = cv2.addWeighted(warped, 0.7, green_overlay, 0.3, 0)

       # Add text with percentage and threshold
       cv2.putText(
           result,
           f"Green: {percentage:.1f}% (T: {threshold}%)",
           (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7,
           (255, 255, 255),
           2
       )

       # Add FPS and elevator status
       cv2.putText(
           result,
           f"FPS: {self.fps:.1f}",
           (10, 60),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7,
           (255, 255, 255),
           2
       )
       
       # Add elevator status based on threshold (FULL at 20% threshold)
       status = "FULL" if threshold <= 20 else "NOT FULL"
       status_color = (0, 0, 255) if status == "FULL" else (0, 255, 0)  # Red for FULL, Green for NOT FULL
       
       cv2.putText(
           result,
           f"Status: {status}",
           (10, 90),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7,
           status_color,
           2
       )

       # Show the result
       cv2.imshow("Processed", result)

       return threshold, warped, mask, percentage


   def update_fps(self):
       """Calculate and update FPS"""
       current_time = time.time()
       self.frame_count += 1


       # Update FPS every second
       time_elapsed = current_time - self.fps_start_time
       if time_elapsed > CONFIG["fps_display_interval"]:
           self.fps = self.frame_count / time_elapsed
           self.frame_count = 0
           self.fps_start_time = current_time


   def save_image_async(self, exact_percentage, frame, threshold):
       """Queue image for asynchronous saving to Google Drive"""
       if CONFIG["enable_threading"]:
           self.image_queue.append((exact_percentage, frame.copy(), threshold))


           # Process the queue in a separate thread if not already running
           if self.upload_thread is None or not self.upload_thread.is_alive():
               import threading
               self.upload_thread = threading.Thread(target=self.process_image_queue)
               self.upload_thread.daemon = True
               self.upload_thread.start()
       else:
           # Synchronous save
           self.save_image(exact_percentage, frame)


   def process_image_queue(self):
       """Process and upload images in the queue"""
       with self.upload_lock:
           # Take only a batch of images to avoid blocking too long
           batch = self.image_queue[:CONFIG["upload_batch_size"]]
           self.image_queue = self.image_queue[CONFIG["upload_batch_size"]:]


       # Process each image in the batch
       for exact_percentage, frame, threshold in batch:
           self.save_image(exact_percentage, frame)


       # If there are more images, schedule another run
       if self.image_queue and self.running:
           import threading
           self.upload_thread = threading.Thread(target=self.process_image_queue)
           self.upload_thread.daemon = True
           self.upload_thread.start()


   def save_image(self, exact_percentage, frame):
       """Save image to Google Drive GREAT folder with threshold subfolders or locally with the specified naming format"""
       global GREAT_FOLDER_ID, THRESHOLD_FOLDER_IDS
       
       # Create filename with the exact format: percentage%-time-date
       current_time = datetime.now()
       formatted_time = current_time.strftime("%H:%M:%S")
       formatted_date = current_time.strftime("%d-%m-%Y")

       # Remove decimal points from percentage to avoid filename issues
       percentage_str = f"{int(exact_percentage)}%"

       # Filename format: 10%-13-47-31-15-03-2025.jpg
       filename = f"{percentage_str}-{formatted_time}-{formatted_date}.jpg"
       filename = filename.replace(":", "-")

       # Determine threshold bucket for saving to the appropriate folder
       threshold = self.get_threshold_bucket(exact_percentage)

       # Use higher JPEG quality (95) for better image quality
       encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

       # Get the warped (calibrated) image
       warped = cv2.warpPerspective(frame, self.perspective_matrix, self.warped_size)

       if self.drive is None or GREAT_FOLDER_ID is None:
           # Save locally if Google Drive is not connected
           local_dir = "saved_images"
           if not os.path.exists(local_dir):
               os.makedirs(local_dir)
               
           # Create GREAT directory if it doesn't exist
           great_dir = os.path.join(local_dir, "GREAT")
           if not os.path.exists(great_dir):
               os.makedirs(great_dir)
               
           # Create threshold directory if it doesn't exist
           threshold_dir = os.path.join(great_dir, f"{threshold}")
           if not os.path.exists(threshold_dir):
               os.makedirs(threshold_dir)
               
           # Save warped image to local directory
           local_path = os.path.join(threshold_dir, filename)
           success = cv2.imwrite(local_path, warped, encode_params)
           
           if success:
               print(f"Calibrated image saved locally: {local_path}")
               return threshold, local_path
           else:
               print("Failed to save calibrated image locally")
               return None, None
       else:
           try:
               # Create a temporary directory if it doesn't exist
               temp_dir = "temp_images"
               if not os.path.exists(temp_dir):
                   os.makedirs(temp_dir)

               # Save warped image temporarily to local disk
               temp_path = os.path.join(temp_dir, filename)
               success = cv2.imwrite(temp_path, warped, encode_params)
               
               if not success:
                   print("Failed to save temporary calibrated image")
                   return None, None

               # Get the folder ID for this threshold
               folder_id = THRESHOLD_FOLDER_IDS.get(threshold)
               
               if folder_id is None:
                   print(f"Warning: No folder ID found for threshold {threshold}, using GREAT folder instead")
                   folder_id = GREAT_FOLDER_ID

               # Upload to Google Drive in the appropriate threshold subfolder
               file_metadata = {
                   'title': filename,
                   'parents': [{'id': folder_id}]
               }
               file_drive = self.drive.CreateFile(file_metadata)
               file_drive.SetContentFile(temp_path)
               file_drive.Upload()

               print(f"Calibrated image uploaded to Google Drive: {filename} in {threshold} subfolder")

               # Store the last uploaded image ID
               self.last_uploaded_image_id = file_drive['id']

               # Clean up temporary file
               os.remove(temp_path)

               return threshold, file_drive['id']

           except Exception as e:
               print(f"Error uploading calibrated image to Google Drive: {e}")
               # Fallback to local save
               local_dir = "saved_images"
               if not os.path.exists(local_dir):
                   os.makedirs(local_dir)
                   
               # Create GREAT directory if it doesn't exist
               great_dir = os.path.join(local_dir, "GREAT")
               if not os.path.exists(great_dir):
                   os.makedirs(great_dir)
                   
               # Create threshold directory if it doesn't exist
               threshold_dir = os.path.join(great_dir, f"{threshold}")
               if not os.path.exists(threshold_dir):
                   os.makedirs(threshold_dir)
                   
               # Save warped image to local directory
               local_path = os.path.join(threshold_dir, filename)
               success = cv2.imwrite(local_path, warped, encode_params)
               
               if success:
                   print(f"Calibrated image saved locally (fallback): {local_path}")
                   return threshold, local_path
               else:
                   print("Failed to save calibrated image locally")
                   return None, None


   def adjust_green_thresholds(self):
       """Interactive adjustment of HSV thresholds for green detection"""
       print("Adjusting green detection thresholds. Press 'q' when done.")


       # Create trackbar window
       cv2.namedWindow("HSV Adjustment")


       # Create trackbars for lower and upper HSV bounds
       cv2.createTrackbar("L-H", "HSV Adjustment", self.lower_green[0], 179, lambda x: None)
       cv2.createTrackbar("L-S", "HSV Adjustment", self.lower_green[1], 255, lambda x: None)
       cv2.createTrackbar("L-V", "HSV Adjustment", self.lower_green[2], 255, lambda x: None)
       cv2.createTrackbar("U-H", "HSV Adjustment", self.upper_green[0], 179, lambda x: None)
       cv2.createTrackbar("U-S", "HSV Adjustment", self.upper_green[1], 255, lambda x: None)
       cv2.createTrackbar("U-V", "HSV Adjustment", self.upper_green[2], 255, lambda x: None)


       # For optimization, process only 15 frames per second during adjustment
       frame_delay = 1 / 15
       last_process_time = 0


       while True:
           current_time = time.time()
           # Skip processing if not enough time has passed
           if current_time - last_process_time < frame_delay:
               # Brief sleep to reduce CPU usage
               time.sleep(0.005)
               if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
               continue


           last_process_time = current_time


           ret, frame = self.camera.read()
           if not ret:
               continue


           # Apply perspective transform if calibrated
           if self.is_calibrated:
               display_frame = cv2.warpPerspective(
                   frame,
                   self.perspective_matrix,
                   (CONFIG["warped_width"], CONFIG["warped_height"])
               )
           else:
               display_frame = frame


           # Get current trackbar positions
           l_h = cv2.getTrackbarPos("L-H", "HSV Adjustment")
           l_s = cv2.getTrackbarPos("L-S", "HSV Adjustment")
           l_v = cv2.getTrackbarPos("L-V", "HSV Adjustment")
           u_h = cv2.getTrackbarPos("U-H", "HSV Adjustment")
           u_s = cv2.getTrackbarPos("U-S", "HSV Adjustment")
           u_v = cv2.getTrackbarPos("U-V", "HSV Adjustment")


           # Create arrays for green detection
           lower_green_temp = np.array([l_h, l_s, l_v])
           upper_green_temp = np.array([u_h, u_s, u_v])


           # Convert to HSV
           hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)


           # Create mask and apply it
           mask = cv2.inRange(hsv, lower_green_temp, upper_green_temp)
           result = cv2.bitwise_and(display_frame, display_frame, mask=mask)


           # Calculate percentage
           total_pixels = mask.shape[0] * mask.shape[1]
           green_pixels = cv2.countNonZero(mask)
           percentage = (green_pixels / total_pixels) * 100


           # Add text with percentage
           cv2.putText(
               result,
               f"Green: {percentage:.1f}%",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7,
               (255, 255, 255),
               2
           )


           # Show images
           cv2.imshow("Original", display_frame)
           cv2.imshow("Mask", mask)
           cv2.imshow("Result", result)


           key = cv2.waitKey(1) & 0xFF
           if key == ord('q'):
               break


       # Update the thresholds with the new values
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
       print("  'u' - Upload current video to Google Drive")
       print("  'q' - Quit")

       # Try to load saved calibration first
       if not self.load_calibration():
           # If no saved calibration, try auto-calibration
           self.auto_calibrate()

       last_frame_time = time.time()
       frame_delay = 1 / CONFIG["target_fps"]  # Target 20 FPS
       print(f"Running with target FPS: {CONFIG['target_fps']}")

       # Pre-allocate memory for frames to reduce memory allocations
       frame_buffer = None

       try:
           while self.running:
               try:
                   # Calculate time since last frame processing
                   current_time = time.time()
                   time_since_last_frame = current_time - last_frame_time

                   # If not enough time has passed, sleep to maintain constant FPS
                   if time_since_last_frame < frame_delay:
                       sleep_time = max(0, frame_delay - time_since_last_frame)
                       time.sleep(sleep_time)

                   # Start timing this frame
                   last_frame_time = time.time()

                   # Process key events first for responsiveness
                   key = cv2.waitKey(1) & 0xFF
                   if key == ord('q'):
                       break
                   elif key == ord('c'):
                       ret, frame = self.camera.read()
                       if ret:
                           try:
                               self.calibrate_frame(frame.copy())
                           except Exception as e:
                               print(f"Error during calibration: {e}")
                               import traceback
                               traceback.print_exc()
                   elif key == ord('a'):
                       self.adjust_green_thresholds()
                   elif key == ord('l'):
                       self.load_calibration()
                   elif key == ord('s'):
                       if self.is_calibrated:
                           ret, frame = self.camera.read()
                           if ret:
                               result = self.process_frame(frame)
                               if result:
                                   _, _, _, exact_percentage = result
                                   self.save_image(exact_percentage, frame)
                   elif key == ord('u'):
                       if self.video_writer and self.video_writer.isOpened():
                           self.video_writer.release()
                           video_path = os.path.join(CONFIG["video_dir"], self.current_video_filename)
                           if os.path.exists(video_path):
                               print(f"Uploading video to Google Drive: {self.current_video_filename}")
                               self.upload_video_to_drive(video_path)
                               # Start a new video segment
                               self.start_new_video_segment()

                   # Read frame
                   ret, frame = self.camera.read()
                   if not ret:
                       print("Failed to read frame from camera.")
                       time.sleep(0.01)
                       continue

                   # Initialize frame buffer if not already done
                   if frame_buffer is None:
                       frame_buffer = np.zeros_like(frame)

                   # Copy frame to buffer to avoid modifying original
                   np.copyto(frame_buffer, frame)

                   # Record raw frame before any processing
                   self.check_video_segment()
                   self.record_frame(frame)

                   # Update FPS counter
                   self.update_fps()

                   if self.is_calibrated:
                       try:
                           result = self.process_frame(frame_buffer)

                           if result:
                               threshold, warped, mask, exact_percentage = result

                               current_time = time.time()
                               time_since_last_save = current_time - self.last_save_time

                               # Save image if threshold changed or enough time has passed
                               if (threshold != self.current_threshold or
                                       time_since_last_save > CONFIG["save_interval"]):

                                   if threshold != self.current_threshold:
                                       print(f"Threshold changed from {self.current_threshold} to {threshold}")

                                   self.save_image_async(exact_percentage, frame.copy(), threshold)
                                   self.current_threshold = threshold
                                   self.last_save_time = current_time
                       except Exception as e:
                           print(f"Error processing frame: {e}")
                   else:
                       # Show original frame with calibration message
                       display_frame = frame_buffer.copy()
                       cv2.putText(
                           display_frame,
                           "NOT CALIBRATED - Press 'c' to calibrate",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (0, 0, 255),
                           2
                       )

                       # Display FPS
                       cv2.putText(
                           display_frame,
                           f"FPS: {self.fps:.1f}",
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (255, 255, 255),
                           2
                       )
                       
                       # Add elevator status (NOT FULL when not calibrated)
                       cv2.putText(
                           display_frame,
                           "Status: NOT FULL",
                           (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (0, 255, 0),  # Green for NOT FULL
                           2
                       )

                       cv2.imshow("Original", display_frame)
               except Exception as e:
                   print(f"Error in main loop: {e}")
                   import traceback
                   traceback.print_exc()
                   time.sleep(0.1)  # Prevent tight error loop

       except KeyboardInterrupt:
           print("\nReceived keyboard interrupt, stopping gracefully...")
       except Exception as e:
           print(f"Fatal error: {e}")
           import traceback
           traceback.print_exc()
       finally:
           # Cleanup
           print("Cleaning up...")
           self.stop_recording()
           
           if CONFIG["enable_threading"] and self.upload_thread and self.upload_thread.is_alive():
               print("Waiting for upload thread to finish...")
               self.upload_thread.join(timeout=3.0)

           if self.camera:
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

