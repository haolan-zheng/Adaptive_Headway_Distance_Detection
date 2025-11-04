# Models

This folder contains the YOLO models used for detection:

- `yolov8n.pt` - Standard YOLOv8 nano model for vehicle detection
- `best.pt` - Custom trained license plate detection model
**Training Details:**
- **Base model:** YOLOv8n
- **Dataset size:** 250+ images (day + night)
- **Dataset source:** Self-collected dashcam footage
- **Training duration:** ~3 hours on M1
- **Image resolution:** 1920*1080

Both models are included for immediate use. No additional setup required!
