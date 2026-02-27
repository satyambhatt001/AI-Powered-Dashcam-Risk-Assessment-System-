import cv2
import numpy as np
import torch
import joblib
import pandas as pd
import playsound  # Sound library
from collections import deque
from ultralytics import YOLO

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

# Load trained risk assessment model and scaler
risk_model = joblib.load("models/risk_assessment_model.pkl")
scaler = joblib.load("models/scaler.pkl")  # Ensure features are scaled correctly

# Define feature columns
feature_columns = ["object_distance", "velocity", "trajectory_angle", "vehicle_speed", "acceleration", "relative_speed"]

# Load input dashcam footage
cap = cv2.VideoCapture("videos/input/dashcam_footage4.mp4")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define vehicle classes for filtering (COCO dataset)
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Store previous frame vehicle speeds for acceleration calculation
prev_speeds = {}

# Track high-risk frequency (store last X high-risk occurrences)
HIGH_RISK_HISTORY = deque(maxlen=30)  # Tracks last 30 frames
HIGH_RISK_THRESHOLD = 10  # If 10 or more high-risk frames in history, play alert

# Alert sound file path
ALERT_SOUND_PATH = "assets/alert.mp3"  # Ensure this file exists

# Function to calculate object distance dynamically
def calculate_distance(bbox, frame_width):
    """Estimate distance based on bounding box size."""
    x1, y1, x2, y2 = bbox
    object_width = x2 - x1
    return max(1, (frame_width / object_width) * 1.5)  # More aggressive scaling

# **Final Risk Classification**
def categorize_risk(object_distance, relative_speed):
    """High-risk now triggers at 18m and 9 km/h."""
    if object_distance > 22:
        return "Low", (0, 255, 0)  # Green
    elif object_distance <= 22 and relative_speed >= 15:
        return "Medium", (0, 255, 255)  # Yellow
    elif object_distance < 21 and relative_speed >= 5:
        return "High", (0, 0, 255)  # Red
    else:
        return "Low", (0, 255, 0)  # Default to low if no extreme case

frame_count = 0  # For frame skipping
FRAME_SKIP = 1  # Process every nth frame (adjustable)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Skip frame processing to improve performance

    # Run object detection on the frame
    results = model(frame)

    high_risk_detected = False  # Flag for tracking high risks in this frame

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = result.names[int(box.cls[0])]   # Object label
            confidence = box.conf[0].item()  # Detection confidence score

            # Only consider vehicles
            if label not in VEHICLE_CLASSES:
                continue

            # Calculate real sensor data
            object_distance = calculate_distance((x1, y1, x2, y2), frame_width)
            velocity = np.random.uniform(5, 30)  # Placeholder for actual speed
            trajectory_angle = np.random.uniform(-45, 45)  # Placeholder
            vehicle_speed = np.random.uniform(20, 80)  # Placeholder

            # Compute acceleration based on previous frame's speed
            prev_speed = prev_speeds.get(label, vehicle_speed)
            acceleration = (vehicle_speed - prev_speed) / (1/fps)  # dv/dt
            
            # Compute relative speed (difference between object and vehicle speed)
            relative_speed = abs(velocity - vehicle_speed)

            # Store current speed for next frame's acceleration calculation
            prev_speeds[label] = vehicle_speed  

            # Prepare features for risk assessment
            features = np.array([[object_distance, velocity, trajectory_angle, vehicle_speed, acceleration, relative_speed]])
            features_scaled = scaler.transform(features)  # Scale using trained scaler

            # Predict risk level
            risk_level = risk_model.predict(features_scaled)[0]

            # **Use updated risk classification**
            risk_text, color = categorize_risk(object_distance, relative_speed)

            # If high risk is detected, set flag
            if risk_text == "High":
                high_risk_detected = True

            # Draw bounding box with risk color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} Risk: {risk_text}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Track high-risk occurrences
    HIGH_RISK_HISTORY.append(1 if high_risk_detected else 0)

    # Check if high-risk alerts should be triggered
    if sum(HIGH_RISK_HISTORY) >= HIGH_RISK_THRESHOLD:
        print("🚨 High-Risk Alert! Playing Sound! 🚨")
        playsound.playsound("C:/Windows/Media/notify.wav", block=False)  # Play sound without blocking execution

    # Display real-time processed video
    cv2.imshow("Real-Time Risk Assessment", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Video processing complete.")
