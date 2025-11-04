#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Adaptive Vehicle Detection and Distance Estimation System
=========================================================

A real-time vehicle detection system using dual YOLO models (license plate + vehicle detection)
with adaptive car width learning for improved distance estimation accuracy.

Features:
- Hybrid detection: License plate (near) + vehicle body (far)
- Adaptive car width learning from detected license plates
- Multi-filter system: oncoming traffic, crossing traffic, distance jumps
- Vehicle tracking with unique IDs
- Minimum 2-second dwell time filter
- Cross-platform GPU support (CUDA for NVIDIA, MPS for Apple Silicon)

Requirements:
- Python 3.8+
- PyTorch (with CUDA support for NVIDIA GPUs)
- ultralytics (YOLOv8)
- OpenCV
- NumPy
- pandas

Author: Haolan Zheng
License: MIT
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# ==================== CONFIGURATION ====================

# Model paths
PLATE_MODEL_PATH = '/Users/haolanzheng/Documents/AI/Plate_Detection/models/training_runs/license_plate_detector_v1/weights/best.pt'
VEHICLE_MODEL_PATH = 'yolov8n.pt'

# Video paths
VIDEO_INPUT = '/Volumes/Haolan/AI_Archive/Plate_Detection/source_videos/day_combined.mp4'
OUTPUT_DIR = Path('/Volumes/Haolan/AI_Archive/Plate_Detection/output_videos')
CSV_DIR = Path('/Volumes/Haolan/AI_Archive/Plate_Detection/output_csv')

# Distance estimation parameters
REAL_PLATE_WIDTH_CM = 30.5
FOCAL_LENGTH_PIXELS = 577
DISTANCE_THRESHOLD = 6.0  # Switch from plate to car detection beyond this distance

# Adaptive car sizing
DEFAULT_CAR_WIDTH_CM = 180
MIN_CAR_WIDTH_CM = 150
MAX_CAR_WIDTH_CM = 220
MAX_WIDTH_HISTORY = 10

# Vehicle tracking
VEHICLE_TIMEOUT_FRAMES = 30
MIN_DWELL_FRAMES = 60  # 2 seconds at 30fps - minimum consecutive frames to display
IOU_THRESHOLD = 0.5

# Filter thresholds
CROSSING_DETECTION_FRAMES = 10
HORIZONTAL_TO_VERTICAL_RATIO = 2.0
MIN_HORIZONTAL_MOVEMENT_PIXELS = 150

DIRECTION_CHECK_FRAMES = 5
SIZE_INCREASE_THRESHOLD = 1.15
Y_MOVEMENT_THRESHOLD = 30
ONCOMING_MIN_DISTANCE = 6.0  # Don't apply oncoming filter if vehicle < 6m

DISTANCE_JUMP_THRESHOLD = 8.0  # Maximum distance change per frame (meters)

# ROI configuration (trapezoid shape for lane focus)
ROI_TOP_LEFT_X_RATIO = 0.44
ROI_TOP_RIGHT_X_RATIO = 0.46
ROI_BOTTOM_LEFT_X_RATIO = 0.33
ROI_BOTTOM_RIGHT_X_RATIO = 0.59
ROI_TOP_Y_RATIO = 0.60
ROI_BOTTOM_Y_RATIO = 0.90

# Multi-vehicle priority
ROI_BOTTOM_PRIORITY_RATIO = 0.7  # Prioritize vehicles in bottom 70% of ROI

# ==================== INITIALIZATION ====================

print("Adaptive Vehicle Detection System")
print("=" * 50)

# Auto-detect best available device
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Device: CUDA (NVIDIA GPU)")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = 'mps'
    print(f"Device: MPS (Apple Silicon GPU)")
else:
    device = 'cpu'
    print(f"Device: CPU (No GPU acceleration)")
    print("Warning: Processing will be slower without GPU acceleration")

# Load models
plate_model = YOLO(PLATE_MODEL_PATH)
vehicle_model = YOLO(VEHICLE_MODEL_PATH)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = str(OUTPUT_DIR / f"vehicle_detection_{timestamp}.mp4")
output_csv_path = str(CSV_DIR / f"distance_data_{timestamp}.csv")

print(f"Input: {VIDEO_INPUT}")
print(f"Output Video: {output_video_path}")
print(f"Output CSV: {output_csv_path}")
print()

# Open video
cap = cv2.VideoCapture(VIDEO_INPUT)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate ROI points
roi_points = np.array([
    [int(frame_width * ROI_TOP_LEFT_X_RATIO), int(frame_height * ROI_TOP_Y_RATIO)],
    [int(frame_width * ROI_TOP_RIGHT_X_RATIO), int(frame_height * ROI_TOP_Y_RATIO)],
    [int(frame_width * ROI_BOTTOM_RIGHT_X_RATIO), int(frame_height * ROI_BOTTOM_Y_RATIO)],
    [int(frame_width * ROI_BOTTOM_LEFT_X_RATIO), int(frame_height * ROI_BOTTOM_Y_RATIO)]
], np.int32)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Video: {frame_width}x{frame_height} @ {fps}fps")
print(f"Total frames: {total_frames}")
print(f"Focal length: {FOCAL_LENGTH_PIXELS}px")
print(f"Minimum dwell time: {MIN_DWELL_FRAMES} frames ({MIN_DWELL_FRAMES/fps:.1f}s)")
print()

# ==================== CLASSES ====================

class DistanceTracker:
    """Tracks vehicle distance and movement patterns over time"""
    
    def __init__(self, max_history=15):
        self.distance_history = deque(maxlen=max_history)
        self.position_history = deque(maxlen=max_history)
        self.x_positions = deque(maxlen=CROSSING_DETECTION_FRAMES)
        self.y_positions = deque(maxlen=CROSSING_DETECTION_FRAMES)
        self.bbox_areas = deque(maxlen=DIRECTION_CHECK_FRAMES)
        self.calibrated_widths = deque(maxlen=MAX_WIDTH_HISTORY)
        self.last_bbox = None
        self.last_distance = None
        self.consecutive_frames = 0
        
    def update(self, distance, position, detected, car_width_cm=None):
        """Update tracker with new detection data"""
        if detected:
            self.distance_history.append(distance)
            self.position_history.append(position)
            self.consecutive_frames += 1
            self.last_bbox = position
            self.last_distance = distance
            
            if position:
                x_center = (position[0] + position[2]) / 2
                y_center = (position[1] + position[3]) / 2
                self.x_positions.append(x_center)
                self.y_positions.append(y_center)
                
                bbox_area = (position[2] - position[0]) * (position[3] - position[1])
                self.bbox_areas.append(bbox_area)
            
            if car_width_cm and MIN_CAR_WIDTH_CM <= car_width_cm <= MAX_CAR_WIDTH_CM:
                self.calibrated_widths.append(car_width_cm)
        else:
            self.consecutive_frames = 0
    
    def check_distance_jump(self, new_distance):
        """Detect abnormal distance changes indicating tracking errors"""
        if self.last_distance is None:
            return False
        return abs(new_distance - self.last_distance) > DISTANCE_JUMP_THRESHOLD
    
    def is_approaching(self):
        """Detect oncoming traffic (size increasing + moving down in frame)"""
        if len(self.bbox_areas) < DIRECTION_CHECK_FRAMES or len(self.y_positions) < DIRECTION_CHECK_FRAMES:
            return False
        
        areas_list = list(self.bbox_areas)
        first_area = areas_list[0]
        last_area = areas_list[-1]
        size_increased = first_area > 0 and (last_area / first_area) > SIZE_INCREASE_THRESHOLD
        
        y_positions_list = list(self.y_positions)
        first_y = y_positions_list[0]
        last_y = y_positions_list[-1]
        moved_down = (last_y - first_y) > Y_MOVEMENT_THRESHOLD
        
        return size_increased and moved_down
    
    def is_crossing_traffic(self):
        """Detect crossing traffic (horizontal movement)"""
        if len(self.x_positions) < CROSSING_DETECTION_FRAMES or len(self.y_positions) < CROSSING_DETECTION_FRAMES:
            return False
        
        x_positions_list = list(self.x_positions)
        y_positions_list = list(self.y_positions)
        
        horizontal_movement = max(x_positions_list) - min(x_positions_list)
        vertical_movement = max(y_positions_list) - min(y_positions_list)
        
        ratio_check = vertical_movement > 0 and horizontal_movement / vertical_movement > HORIZONTAL_TO_VERTICAL_RATIO
        absolute_check = horizontal_movement > MIN_HORIZONTAL_MOVEMENT_PIXELS
        
        return ratio_check and absolute_check
    
    def get_smoothed_distance(self):
        """Calculate exponentially weighted average of distance history"""
        if len(self.distance_history) == 0:
            return None
        weights = np.exp(np.linspace(-1, 0, len(self.distance_history)))
        weights /= weights.sum()
        return np.average(list(self.distance_history), weights=weights)
    
    def get_calibrated_car_width(self):
        """Get learned car width or use default"""
        if len(self.calibrated_widths) > 0:
            return np.median(list(self.calibrated_widths))
        return DEFAULT_CAR_WIDTH_CM

# ==================== HELPER FUNCTIONS ====================

def check_in_roi(x_center, y_center):
    """Check if point is inside trapezoid ROI"""
    roi_top_y = int(frame_height * ROI_TOP_Y_RATIO)
    roi_bottom_y = int(frame_height * ROI_BOTTOM_Y_RATIO)
    roi_top_left_x = int(frame_width * ROI_TOP_LEFT_X_RATIO)
    roi_top_right_x = int(frame_width * ROI_TOP_RIGHT_X_RATIO)
    roi_bottom_left_x = int(frame_width * ROI_BOTTOM_LEFT_X_RATIO)
    roi_bottom_right_x = int(frame_width * ROI_BOTTOM_RIGHT_X_RATIO)
    
    if roi_top_y <= y_center <= roi_bottom_y:
        ratio = (y_center - roi_top_y) / (roi_bottom_y - roi_top_y)
        left_bound = roi_top_left_x + ratio * (roi_bottom_left_x - roi_top_left_x)
        right_bound = roi_top_right_x + ratio * (roi_bottom_right_x - roi_top_right_x)
        return left_bound <= x_center <= right_bound
    return False

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union for two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# ==================== MAIN PROCESSING LOOP ====================

tracked_vehicles = {}
next_vehicle_id = 1
current_vehicle_id = None
last_seen_frame = {}
frame_count = 0

# CSV data storage
csv_data = []

print("Processing video...")
print("-" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Progress indicator
    if frame_count % 500 == 0:
        progress = frame_count / total_frames * 100
        vehicle_status = f"Vehicle #{current_vehicle_id}" if current_vehicle_id else "No vehicle"
        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) | {vehicle_status}")
    
    display_frame = frame.copy()
    cv2.polylines(display_frame, [roi_points], True, (255, 255, 0), 2)
    
    # Timeout check - remove vehicles not seen recently
    vehicles_to_remove = []
    for vid in list(tracked_vehicles.keys()):
        if frame_count - last_seen_frame.get(vid, 0) > VEHICLE_TIMEOUT_FRAMES:
            vehicles_to_remove.append(vid)
            if vid == current_vehicle_id:
                current_vehicle_id = None
    
    for vid in vehicles_to_remove:
        del tracked_vehicles[vid]
        if vid in last_seen_frame:
            del last_seen_frame[vid]
    
    # STEP 1: Detect license plates
    plate_results = plate_model(frame, conf=0.20, verbose=False, device=device)
    
    plate_detected = False
    plate_distance = None
    plate_box = None
    
    for box in plate_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        if check_in_roi(x_center, y_center):
            plate_width_pixels = abs(x2 - x1)
            if plate_width_pixels > 0:
                distance_cm = (REAL_PLATE_WIDTH_CM * FOCAL_LENGTH_PIXELS) / plate_width_pixels
                plate_distance = distance_cm / 100
                plate_box = (x1, y1, x2, y2)
                plate_detected = True
                break
    
    # STEP 2: Detect vehicles in ROI
    vehicle_results = vehicle_model(frame, classes=[2], conf=0.30, verbose=False, device=device)
    
    vehicle_detections = []
    for box in vehicle_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        if check_in_roi(x_center, y_center):
            car_width_px = abs(x2 - x1)
            bbox_area = car_width_px * abs(y2 - y1)
            
            vehicle_detections.append({
                'bbox': (x1, y1, x2, y2),
                'area': bbox_area,
                'y_center': y_center,
                'width_pixels': car_width_px
            })
    
    # STEP 2.1: For learning phase - detect full car width in entire frame
    full_frame_car_for_learning = None
    
    if plate_detected and plate_distance < DISTANCE_THRESHOLD:
        for box in vehicle_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            plate_x1, plate_y1, plate_x2, plate_y2 = plate_box
            plate_center_x = (plate_x1 + plate_x2) / 2
            plate_center_y = (plate_y1 + plate_y2) / 2
            
            if x1 <= plate_center_x <= x2 and y1 <= plate_center_y <= y2:
                full_car_width_pixels = abs(x2 - x1)
                full_frame_car_for_learning = {
                    'bbox': (x1, y1, x2, y2),
                    'width_pixels': full_car_width_pixels
                }
                break
    
    # STEP 2.5: Prioritize vehicles in bottom 70% of ROI + largest bbox
    vehicle_box = None
    vehicle_detected = False
    car_width_pixels = 0
    
    if vehicle_detections:
        roi_height = int(frame_height * ROI_BOTTOM_Y_RATIO) - int(frame_height * ROI_TOP_Y_RATIO)
        min_y_threshold = int(frame_height * ROI_TOP_Y_RATIO) + (roi_height * (1 - ROI_BOTTOM_PRIORITY_RATIO))
        
        priority_vehicles = [v for v in vehicle_detections if v['y_center'] >= min_y_threshold]
        
        if not priority_vehicles:
            priority_vehicles = vehicle_detections
        
        best_vehicle = max(priority_vehicles, key=lambda v: v['area'])
        vehicle_box = best_vehicle['bbox']
        car_width_pixels = best_vehicle['width_pixels']
        vehicle_detected = True
    
    # STEP 3: Match with existing vehicles using IoU
    matched_vehicle_id = None
    
    if vehicle_detected:
        best_iou = 0
        best_match_id = None
        
        for vid, vtracker in tracked_vehicles.items():
            if vtracker.last_bbox is not None:
                iou = calculate_iou(vehicle_box, vtracker.last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = vid
        
        if best_iou > IOU_THRESHOLD:
            matched_vehicle_id = best_match_id
            current_vehicle_id = best_match_id
        else:
            # Graceful handoff: if only one vehicle and recently seen, continue tracking
            if len(tracked_vehicles) == 1:
                last_vehicle_id = list(tracked_vehicles.keys())[0]
                frames_since_last = frame_count - last_seen_frame[last_vehicle_id]
                
                if frames_since_last < 90:
                    matched_vehicle_id = last_vehicle_id
                    current_vehicle_id = last_vehicle_id
                else:
                    matched_vehicle_id = next_vehicle_id
                    tracked_vehicles[matched_vehicle_id] = DistanceTracker()
                    next_vehicle_id += 1
                    current_vehicle_id = matched_vehicle_id
            else:
                matched_vehicle_id = next_vehicle_id
                tracked_vehicles[matched_vehicle_id] = DistanceTracker()
                next_vehicle_id += 1
                current_vehicle_id = matched_vehicle_id
    
    # STEP 4: Calculate distance and update tracker
    if matched_vehicle_id:
        last_seen_frame[matched_vehicle_id] = frame_count
        tracker = tracked_vehicles[matched_vehicle_id]
        tracker.last_bbox = vehicle_box
        
        final_distance = None
        final_box = None
        detection_method = ""
        learned_car_width = None
        
        calibrated_car_width = tracker.get_calibrated_car_width()
        
        # Hybrid detection logic
        if plate_detected and plate_distance < DISTANCE_THRESHOLD:
            final_distance = plate_distance
            final_box = plate_box
            detection_method = f"PLATE {plate_distance:.1f}m"
            
            # Learn car width from full frame detection if available
            if full_frame_car_for_learning:
                full_car_width_pixels = full_frame_car_for_learning['width_pixels']
                learned_car_width = (full_car_width_pixels * plate_distance * 100) / FOCAL_LENGTH_PIXELS
                if MIN_CAR_WIDTH_CM <= learned_car_width <= MAX_CAR_WIDTH_CM:
                    detection_method = f"PLATE {plate_distance:.1f}m [LEARNING]"
            elif vehicle_detected and car_width_pixels > 0:
                learned_car_width = (car_width_pixels * plate_distance * 100) / FOCAL_LENGTH_PIXELS
                if MIN_CAR_WIDTH_CM <= learned_car_width <= MAX_CAR_WIDTH_CM:
                    detection_method = f"PLATE {plate_distance:.1f}m [LEARNING]"
            
        elif plate_detected and plate_distance >= DISTANCE_THRESHOLD:
            if vehicle_detected and car_width_pixels > 0:
                vehicle_distance = (calibrated_car_width * FOCAL_LENGTH_PIXELS) / car_width_pixels / 100
                final_distance = vehicle_distance
                final_box = vehicle_box
                width_status = "LEARNED" if len(tracker.calibrated_widths) > 0 else "DEFAULT"
                detection_method = f"CAR {vehicle_distance:.1f}m [{width_status}: {calibrated_car_width:.0f}cm]"
            else:
                final_distance = plate_distance
                final_box = plate_box
                detection_method = f"PLATE {plate_distance:.1f}m (far)"
            
        elif not plate_detected and vehicle_detected and car_width_pixels > 0:
            vehicle_distance = (calibrated_car_width * FOCAL_LENGTH_PIXELS) / car_width_pixels / 100
            final_distance = vehicle_distance
            final_box = vehicle_box
            width_status = "LEARNED" if len(tracker.calibrated_widths) > 0 else "DEFAULT"
            detection_method = f"CAR {vehicle_distance:.1f}m [{width_status}: {calibrated_car_width:.0f}cm]"
        
        # Check for distance jump before updating
        if final_distance and tracker.check_distance_jump(final_distance):
            del tracked_vehicles[matched_vehicle_id]
            if matched_vehicle_id in last_seen_frame:
                del last_seen_frame[matched_vehicle_id]
            if matched_vehicle_id == current_vehicle_id:
                current_vehicle_id = None
            matched_vehicle_id = None
        elif final_distance:
            tracker.update(final_distance, final_box, True, learned_car_width)
        else:
            tracker.update(None, None, True, None)
        
        # Apply filters
        if matched_vehicle_id and final_distance and final_distance > ONCOMING_MIN_DISTANCE and tracker.is_approaching():
            del tracked_vehicles[matched_vehicle_id]
            if matched_vehicle_id in last_seen_frame:
                del last_seen_frame[matched_vehicle_id]
            if matched_vehicle_id == current_vehicle_id:
                current_vehicle_id = None
            matched_vehicle_id = None
        
        elif matched_vehicle_id and tracker.is_crossing_traffic():
            del tracked_vehicles[matched_vehicle_id]
            if matched_vehicle_id in last_seen_frame:
                del last_seen_frame[matched_vehicle_id]
            if matched_vehicle_id == current_vehicle_id:
                current_vehicle_id = None
            matched_vehicle_id = None
        
        # Draw vehicle if minimum dwell time reached
        if matched_vehicle_id:
            smoothed_distance = tracker.get_smoothed_distance()
            
            if smoothed_distance and tracker.consecutive_frames >= MIN_DWELL_FRAMES and final_box:
                x1, y1, x2, y2 = final_box
                
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                
                vehicle_label = f"Vehicle #{matched_vehicle_id}"
                distance_label = f"{smoothed_distance:.1f}m"
                
                cv2.putText(display_frame, vehicle_label, (int(x1), int(y1)-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, distance_label, (int(x1), int(y1)-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, detection_method, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Export to CSV
                csv_data.append({
                    'frame': frame_count,
                    'timestamp_sec': frame_count / fps,
                    'vehicle_id': matched_vehicle_id,
                    'distance_m': smoothed_distance,
                    'detection_method': detection_method.split()[0],  # PLATE or CAR
                    'consecutive_frames': tracker.consecutive_frames
                })
    
    # Status overlay
    status = f"Frame: {frame_count} | Device: {device.upper()}"
    cv2.putText(display_frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    out.write(display_frame)
    frame_count += 1

cap.release()
out.release()

# Export CSV data
if csv_data:
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False)
    print("-" * 50)
    print("Processing complete!")
    print(f"Output video: {output_video_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Total vehicles tracked: {next_vehicle_id - 1}")
    print(f"CSV records exported: {len(csv_data)}")
else:
    print("-" * 50)
    print("Processing complete!")
    print(f"Output video: {output_video_path}")
    print(f"Total vehicles tracked: {next_vehicle_id - 1}")
    print("Warning: No CSV data exported (no vehicles met minimum dwell time)")

