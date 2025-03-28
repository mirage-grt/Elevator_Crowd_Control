# Carpet Monitoring System

A sophisticated real-time monitoring system designed for elevator carpets that detects occupancy and records video footage. The system utilizes advanced computer vision techniques to analyze carpet coverage and automatically manages recordings through Google Drive integration.

## Features

### Core Functionality
- Real-time video monitoring with live feed display
- Automatic green area detection using HSV color space
- Intelligent threshold-based occupancy detection
- Video recording with automatic 2-minute segmentation
- Google Drive integration for secure video storage
- Web interface for remote monitoring and control
- CLI commands for system management
- Advanced calibration system for precise detection

### Technical Features
- Multi-threaded architecture for optimal performance
- Automatic video segment management
- Timestamp overlay on recorded videos
- FPS monitoring and display
- Automatic threshold bucket categorization
- Persistent calibration data storage
- Fallback local storage when cloud upload fails
- Error handling and recovery mechanisms

## Prerequisites

### Hardware Requirements
- Camera (USB webcam or Raspberry Pi camera module)
- Sufficient storage space for temporary video files
- Internet connection for Google Drive integration

### Software Requirements
- Python 3.x
- OpenCV (for computer vision processing)
- Flask (for web interface)
- Google Drive API credentials
- Required Python packages:
  ```
  pip install opencv-python numpy flask pydrive oauth2client
  ```

## Setup

### 1. Repository Setup
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Dependencies Installation
```bash
pip install -r requirements.txt
```

### 3. Google Drive API Setup
1. Create a Google Cloud Project
2. Enable the Google Drive API
3. Create OAuth 2.0 credentials
4. Download the credentials file
5. Place the credentials file in the project directory
6. First run will prompt for Google Drive authentication

### 4. Directory Structure
```
project/
├── web_interface.py
├── rpi_camera_integration.py
├── templates/
│   └── index.html
├── saved_images/
│   └── GREAT/
│       └── video/
├── temp_images/
└── calibration_points.npy
```

## Usage

### Starting the System

1. Launch the web interface:
   ```bash
   python web_interface.py
   ```

2. Access the interface:
   - Local: http://localhost:5000
   - Network: http://[your-ip]:5000

### Authentication
- Admin Account:
  - Username: `admin`
  - Password: `admin`
- User Account:
  - Username: `user`
  - Password: `user`

### Available Commands

#### CLI Commands
- `start` - Initialize and start the monitoring system
- `stop` - Gracefully stop the monitor and upload current video
- `calibrate` - Launch the calibration process
- `status` - Display current system status

#### Keyboard Shortcuts
- `c` - Manual calibration mode
- `a` - Adjust green color detection thresholds
- `l` - Load previously saved calibration
- `s` - Save current frame as image
- `u` - Upload current video segment to Google Drive
- `q` - Quit the application

### Calibration Process

1. **Manual Calibration**
   - Click "Calibrate" in the web interface or type `calibrate` in CLI
   - Click on the four corners of the carpet area in clockwise order
   - System will validate and save calibration points

2. **Auto-Calibration**
   - System attempts automatic calibration on startup
   - Requires clear view of green carpet area
   - Minimum area threshold must be met

### Video Recording

#### Recording Features
- Automatic 2-minute segment creation
- Timestamp overlay on each frame
- Automatic upload to Google Drive
- Local storage fallback if upload fails
- Organized storage in Google Drive folders

#### Video Management
- Videos are stored in the "video" folder in Google Drive
- Each segment is named with date and time
- Automatic cleanup of local files after successful upload
- Error handling for failed uploads

## Configuration

### System Parameters
```python
CONFIG = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "save_interval": 10,
    "min_area_for_calibration": 5000,
    "warped_width": 400,
    "warped_height": 300,
    "fps_display_interval": 1.0,
    "upload_batch_size": 5,
    "enable_threading": True,
    "target_fps": 20,
    "video_segment_duration": 120,
    "video_fps": 30
}
```

### Threshold Buckets
- System categorizes green area percentage into buckets:
  - 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
- Elevator considered "FULL" at 20% threshold or below

## Troubleshooting

### Common Issues
1. Camera not detected
   - Check camera connection
   - Verify camera permissions
   - Try different camera index

2. Calibration failures
   - Ensure proper lighting
   - Check for clear view of carpet
   - Verify corner selection accuracy

3. Upload failures
   - Check internet connection
   - Verify Google Drive credentials
   - Check available storage space

### Error Messages
- Detailed error logging in console
- Fallback mechanisms for critical operations
- Automatic retry for failed uploads

## Maintenance

### Regular Tasks
- Monitor disk space usage
- Check Google Drive storage
- Verify camera alignment
- Test calibration accuracy

### Backup Procedures
- Regular calibration point backups
- Local storage management
- Google Drive folder organization

## Security Considerations

### Access Control
- Web interface authentication
- Secure credential storage
- API key protection

### Data Protection
- Local file cleanup
- Secure upload protocols
- Credential encryption

## Support

For technical support or questions:
1. Check the troubleshooting guide
2. Review error logs
3. Contact system administrator

## License

[APACHE 2.0]

## Contributing

[Be respectful and open to collaboration.

Avoid spam or irrelevant contributions.

Follow best practices for security and performance.] 