# AI-Powered Dashcam Risk Assessment System

A real-time computer vision system that analyzes dashcam footage to detect road hazards and risky driving behaviors. The system generates dynamic risk scores and visual warnings to improve road safety using AI.

---

## Overview

This project combines YOLOv8 object detection with a Decision Tree risk assessment model to analyze real-time video streams and simulated sensor data.

It identifies high-risk objects (vehicles, pedestrians, obstacles) and calculates a contextual risk level to assist drivers or fleet monitoring systems.

---

## Features

- Object detection using YOLOv8
- Risk level assessment using a trained Decision Tree model
- Real-time video processing
- Simulated sensor data integration
- Visual alerts for high-risk objects
- Dynamic risk scoring

---

## Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Scikit-learn
- NumPy
- Pandas

---

## Project Structure

```
AI-Powered-Dashcam-Risk-Assessment-System-
│
├── src/
│   ├── real_time_risk.py
│   ├── risk_assessment.py
│   └── utils/
│
├── models/
├── videos/
├── data/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/satyambhatt001/AI-Powered-Dashcam-Risk-Assessment-System-.git
cd AI-Powered-Dashcam-Risk-Assessment-System-
```

### 2. Create Virtual Environment

```bash
python -m venv risk-assessment-env
source risk-assessment-env/bin/activate   # macOS/Linux
risk-assessment-env\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

### Verify Installation

```bash
python src/utils/verify_installation.py
```

### Simulate Sensor Data

```bash
python src/utils/simulate_sensor_data.py
```

### Train Risk Model

```bash
python src/risk_assessment.py
```

### Run Real-Time Risk Detection

```bash
python src/real_time_risk.py
```

---

## Changing Input Video

1. Place a new video inside the `videos/` directory  
2. Update the video path in `real_time_risk.py`

---

## Future Improvements

- Real-time deployment on edge devices
- Deep learning-based risk prediction model
- Integration with IoT sensors
- Dashboard for fleet analytics

---

## License

This project is developed for academic and research purposes.

---

## Author

Satyam Bhatt  
AI and Machine Learning Enthusiast
