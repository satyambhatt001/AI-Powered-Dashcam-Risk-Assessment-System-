import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sklearn.tree import DecisionTreeClassifier
from playsound import playsound
import matplotlib.pyplot as plt

def verify_libraries():
    """Verify the installation of required Python libraries."""
    try:
        print(f"OpenCV Installed: {cv2.__version__}")
        print(f"NumPy Installed: {np.__version__}")
        print(f"Pandas Installed: {pd.__version__}")
        print(f"PyTorch Installed: {torch.__version__}")
        print(f"scikit-learn Installed: {DecisionTreeClassifier().__class__.__name__}")
    except Exception as e:
        print(f"Error verifying libraries: {e}")

def verify_yolo():
    """Check if the YOLOv8 model can be loaded successfully."""
    try:
        model = YOLO("yolov8n.pt")
        print("YOLO Model Loaded Successfully")
    except Exception as e:
        print(f"YOLO Model Loading Failed: {e}")

def verify_audio():
    """Verify the functionality of the playsound module."""
    try:
        playsound("C:/Windows/Media/notify.wav")  # Windows default notification sound
        print("Playsound Module Working")
    except Exception as e:
        print(f"Playsound Module Error: {e}")

if __name__ == "__main__":
    print("Verifying Installation of Required Dependencies...\n")
    verify_libraries()
    verify_yolo()
    verify_audio()
    print("\nVerification Complete.")
