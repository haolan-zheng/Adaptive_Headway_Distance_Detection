# Adaptive Headway Distance Detection

A real-time vehicle detection and distance estimation system using dual YOLO models with adaptive car width learning for improved accuracy.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)

## Features

- 🚗 **Dual YOLO Detection**: License plate detection (near) + vehicle body detection (far)
- 📏 **Adaptive Learning**: Automatically learns and calibrates car widths for improved distance estimation
- 🎯 **Smart Filtering**: Multi-filter system to remove oncoming traffic, crossing traffic, and tracking errors
- 🔢 **Vehicle Tracking**: Assigns unique IDs and tracks vehicles across frames
- ⏱️ **Minimum Dwell Time**: 2-second filter to eliminate brief false detections
- 🖥️ **Cross-Platform GPU**: Automatic GPU acceleration (CUDA for NVIDIA, MPS for Apple Silicon)
- 📊 **CSV Export**: Frame-by-frame distance data export for analysis


## Requirements

- Python 3.8+
- PyTorch (with CUDA support for NVIDIA GPUs)
- ultralytics (YOLOv8)
- OpenCV
- NumPy
- pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Adaptive_Headway_Distance_Detection.git
cd Adaptive_Headway_Distance_Detection
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install ultralytics opencv-python numpy pandas
```

3. Download YOLO models:
   - YOLOv8n (vehicle detection): Auto-downloaded on first run
   - Custom license plate model: Train your own or use pre-trained

## Usage

1. Update file paths in the configuration section:
```python
VIDEO_INPUT = 'path/to/your/video.mp4'
OUTPUT_DIR = Path('path/to/output/videos')
CSV_DIR = Path('path/to/output/csv')
PLATE_MODEL_PATH = 'path/to/license_plate_model.pt'
```

2. Run the script:
```bash
python Adaptive_Headway_Distance_Detection.py
```

3. Output files:
   - Video: `output_videos/vehicle_detection_YYYYMMDD_HHMMSS.mp4`
   - CSV: `output_csv/distance_data_YYYYMMDD_HHMMSS.csv`

## Configuration

Key parameters you can adjust:
```python
# Distance estimation
REAL_PLATE_WIDTH_CM = 30.5  # Standard license plate width
FOCAL_LENGTH_PIXELS = 577   # Camera-specific (please calibrate for your camera)
# In addition, my dashcam is a wide angle one, so any vehicles that are beyond 25 meters are really small in the footage. If you want to capture more vehicles that are afar, I recommend using a dual camera system with one wide angle and a telephoto lens combined. 

# Detection thresholds
MIN_DWELL_FRAMES = 60       # Minimum frames to display (2 sec at 30fps)
DISTANCE_JUMP_THRESHOLD = 8.0  # Max distance change per frame (meters)

# ROI (Region of Interest) - adjust for your camera angle
ROI_TOP_Y_RATIO = 0.60
ROI_BOTTOM_Y_RATIO = 0.90
```

## How It Works

### Hybrid Detection System
1. **Close range (< 6m)**: Uses license plate detection for high accuracy
2. **Far range (> 6m)**: Uses vehicle body detection with learned car width
3. **Adaptive learning**: System learns actual car widths from plate detections

### Distance Estimation
Uses pinhole camera model:
```
distance = (real_width × focal_length) / pixel_width
```

### Filtering Pipeline
- **Oncoming traffic filter**: Detects vehicles moving toward camera
- **Crossing traffic filter**: Detects lateral movement patterns
- **Distance jump filter**: Catches tracking errors (>8m changes)
- **Minimum dwell time**: Requires 2 seconds of continuous detection

## Performance

Tested on MacBook Air M1 (2020):
- Processing speed: ~0.5x real-time (10 min video in 20 min)
- Power consumption: 15-20W

I have never tested the script on a Windows platform with RTX graphic cards, but the script should be able to recognize your device and utilize the local GPU for acclerated processing. 

## Known Limitations

- Intersection scenarios with stopped traffic may produce false detections
- Requires camera calibration (focal length) for accurate distance estimation
- Performance varies with video quality and lighting conditions. Significant sun glare may interfere with the vehicle/plate detection. 
- License plate model must be trained separately

## CSV Output Format

| Column | Description |
|--------|-------------|
| frame | Frame number in video |
| timestamp_sec | Time in seconds from start |
| vehicle_id | Unique vehicle identifier |
| distance_m | Estimated distance in meters |
| detection_method | PLATE or CAR |
| consecutive_frames | Tracking duration |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Haolan Zheng**

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team
```

4. Click **"Commit changes"**

---

### **Step 5: Add requirements.txt**

1. Click **"Add file"** → **"Create new file"**
2. Name it: `requirements.txt`
3. Add content:
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
```
4. Click **"Commit new file"**

---

### **Step 6: Add a .gitignore** (if you didn't check it during creation)

1. Click **"Add file"** → **"Create new file"**
2. Name it: `.gitignore`
3. Add content:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Output files
*.mp4
*.avi
*.csv

# Models (too large for git)
*.pt
*.pth
models/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
