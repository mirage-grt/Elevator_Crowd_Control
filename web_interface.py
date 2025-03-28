from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
import threading
import queue
import time
import sys
import io
from datetime import datetime
from rpi_camera_integration import CarpetMonitor, CONFIG
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Global variables
monitor = None
output_queue = queue.Queue()
frame_queue = queue.Queue()
is_running = False

# Custom stdout handler to capture terminal output
class OutputHandler:
    def __init__(self, queue):
        self.queue = queue
        self.buffer = io.StringIO()

    def write(self, text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        if text.strip():  # Only queue non-empty lines
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.queue.put(f"[{timestamp}] {text.strip()}")
            self.buffer.write(text)

    def flush(self):
        self.buffer.flush()

# Redirect stdout to our custom handler
sys.stdout = OutputHandler(output_queue)

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            # Resize frame for web display if needed
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)  # Reduced delay for smoother video

@app.route('/')
def index():
    if 'role' not in session:
        return redirect(url_for('login', role='user'))
    return redirect(url_for('user_dashboard'))

@app.route('/admin')
def admin():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    return render_template('admin.html', local_ip=get_local_ip())

@app.route('/user')
def user_dashboard():
    if 'role' not in session:
        return redirect(url_for('login', role='user'))
    return render_template('user.html', local_ip=get_local_ip())

@app.route('/login/<role>', methods=['GET', 'POST'])
def login(role):
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Debug output
        sys.stdout.write(f"Login attempt - Role: {role}, Username: {username}\n")
        
        # Simple authentication (replace with proper authentication)
        if role == 'admin':
            if username == 'admin' and password == 'admin':
                session['role'] = 'admin'
                sys.stdout.write("Admin login successful\n")
                return redirect(url_for('admin'))
            else:
                sys.stdout.write("Admin login failed - Invalid credentials\n")
        elif role == 'user':
            if username == 'user' and password == 'user':
                session['role'] = 'user'
                sys.stdout.write("User login successful\n")
                return redirect(url_for('user_dashboard'))
            else:
                sys.stdout.write("User login failed - Invalid credentials\n")
        
        return render_template('login.html', role=role, error='Invalid credentials', local_ip=get_local_ip())
    
    # Clear any existing session when showing login page
    session.clear()
    return render_template('login.html', role=role, local_ip=get_local_ip())

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login', role='user'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    global monitor
    if monitor is None:
        return jsonify({
            'running': False,
            'calibrated': False,
            'current_threshold': None,
            'fps': 0,
            'status': 'NOT RUNNING'
        })
    
    return jsonify({
        'running': monitor.running,
        'calibrated': monitor.is_calibrated,
        'current_threshold': monitor.current_threshold,
        'fps': round(monitor.fps, 1),
        'status': 'FULL' if monitor.current_threshold and monitor.current_threshold <= 20 else 'NOT FULL'
    })

@app.route('/cli_output')
def get_cli_output():
    outputs = []
    while not output_queue.empty():
        outputs.append(output_queue.get())
    return jsonify(outputs)

@app.route('/cli_command', methods=['POST'])
def send_cli_command():
    command = request.json.get('command', '').strip()
    if command:
        # Process command and add to output queue
        if command.lower() == 'start':
            start_monitor()
            sys.stdout.write("Starting monitor...\n")
        elif command.lower() == 'stop':
            stop_monitor()
            sys.stdout.write("Stopping monitor...\n")
        elif command.lower() == 'calibrate':
            if monitor:
                sys.stdout.write("Starting calibration...\n")
                sys.stdout.write("Click on the four corners of the carpet area (clockwise from top-left)\n")
                sys.stdout.write("Press ESC to cancel\n")
                # Get a frame and start calibration
                ret, frame = monitor.camera.read()
                if ret:
                    monitor.calibrate_frame(frame)
                else:
                    sys.stdout.write("Failed to read frame for calibration\n")
        elif command.lower() == 'status':
            if monitor:
                sys.stdout.write(f"Status: Running={monitor.running}, Calibrated={monitor.is_calibrated}, "
                               f"Threshold={monitor.current_threshold}%, FPS={monitor.fps:.1f}\n")
        else:
            sys.stdout.write(f"Unknown command: {command}\n")
    return "Command processed"

def start_monitor():
    global monitor, is_running
    if not is_running:
        try:
            monitor = CarpetMonitor()
            is_running = True
            monitor.running = True
            
            # Try to load saved calibration points
            if os.path.exists("calibration_points.npy"):
                sys.stdout.write("Loading saved calibration points...\n")
                monitor.load_calibration()
            
            threading.Thread(target=monitor_thread, daemon=True).start()
            sys.stdout.write("Monitor thread started successfully\n")
        except Exception as e:
            sys.stdout.write(f"Error starting monitor: {str(e)}\n")
            import traceback
            traceback.print_exc()
            is_running = False
            monitor = None

def stop_monitor():
    global monitor, is_running
    try:
        is_running = False
        if monitor:
            # First stop the monitor's main loop
            monitor.running = False
            
            # Stop video recording and upload
            if monitor.video_writer and monitor.video_writer.isOpened():
                print("Stopping video recording and uploading...")
                monitor.video_writer.release()
                video_path = os.path.join(monitor.CONFIG["video_dir"], monitor.current_video_filename)
                if os.path.exists(video_path):
                    print(f"Uploading video to Google Drive: {monitor.current_video_filename}")
                    monitor.upload_video_to_drive(video_path)
            
            # Clean up resources
            if monitor.camera:
                monitor.camera.release()
            cv2.destroyAllWindows()
            
            # Wait for any upload threads to finish
            if CONFIG["enable_threading"] and monitor.upload_thread and monitor.upload_thread.is_alive():
                print("Waiting for upload thread to finish...")
                monitor.upload_thread.join(timeout=3.0)
            
            monitor = None
            print("Monitor stopped successfully")
    except Exception as e:
        print(f"Error stopping monitor: {e}")
        import traceback
        traceback.print_exc()

def monitor_thread():
    global monitor
    try:
        while is_running and monitor:
            ret, frame = monitor.camera.read()
            if ret:
                # Always send the raw frame to web interface
                frame_queue.put(frame)
                
                # Process frame if calibrated
                if monitor.is_calibrated:
                    result = monitor.process_frame(frame)
                    if result:
                        threshold, warped, mask, exact_percentage = result
                        # Ensure all information is visible on the processed frame
                        display_frame = warped.copy()
                        # Add text with percentage and threshold
                        cv2.putText(
                            display_frame,
                            f"Green: {exact_percentage:.1f}% (T: {threshold}%)",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )
                        # Add FPS
                        cv2.putText(
                            display_frame,
                            f"FPS: {monitor.fps:.1f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )
                        # Add elevator status
                        status = "FULL" if threshold <= 20 else "NOT FULL"
                        status_color = (0, 0, 255) if status == "FULL" else (0, 255, 0)
                        cv2.putText(
                            display_frame,
                            f"Status: {status}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            status_color,
                            2
                        )
                        # Show processed frame in OpenCV window
                        cv2.imshow("Processed", display_frame)
                        if threshold != monitor.current_threshold:
                            monitor.save_image_async(exact_percentage, frame.copy(), threshold)
                            monitor.current_threshold = threshold
                else:
                    # Show original frame with calibration message in OpenCV window
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
                    # Show FPS even when not calibrated
                    cv2.putText(
                        display_frame,
                        f"FPS: {monitor.fps:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    # Show uncalibrated frame in OpenCV window
                    cv2.imshow("Original", display_frame)
            
            # Update FPS
            monitor.update_fps()
            # Handle OpenCV window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                ret, frame = monitor.camera.read()
                if ret:
                    monitor.calibrate_frame(frame)
            time.sleep(0.05)  # 20 FPS
            
    except Exception as e:
        sys.stdout.write(f"Error in monitor thread: {str(e)}\n")
        import traceback
        traceback.print_exc()
        time.sleep(1)

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

if __name__ == '__main__':
    print("Starting web interface...")
    print(f"Local access: http://localhost:5000")
    print(f"Network access: http://{get_local_ip()}:5000")
    app.run(host='0.0.0.0', port=5000, debug=False) 