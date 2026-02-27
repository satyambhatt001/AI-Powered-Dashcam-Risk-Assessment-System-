# AI-Powered-Dashcam-Risk-Assessment-System

A real-time computer vision system that analyzes dashcam footage to detect road hazards and risky driving behaviors, generating instant alerts and dynamic risk scores to improve driving safety.

---

## Features

- Object detection using **YOLOv8**
- Risk level assessment using a trained **Decision Tree model**
- Real-time video processing with simulated sensor data
- Visual warnings for high-risk objects

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/satyambhatt001/AI-Powered-Dashcam-Risk-Assessment-System-.git
cd AI-Powered-Dashcam-Risk-Assessment-System-
```

### 2. Set Up a Virtual Environment

```bash
python -m venv risk-assessment-env
```

Activate the environment:

**macOS/Linux**
```bash
source risk-assessment-env/bin/activate
```

**Windows**
```bash
risk-assessment-env\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
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

### Train the Risk Assessment Model

```bash
python src/risk_assessment.py
```

### Run Object Detection & Risk Assessment on Video

```bash
python src/real_time_risk.py
```

---

## Changing the Input Video

1. Save your new video inside the `videos` directory.
2. Update the video path inside `real_time_risk.py` accordingly.
