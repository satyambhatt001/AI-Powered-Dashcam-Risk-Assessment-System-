# AI-Powered Risk Assessment System  

This project is an AI-powered risk assessment and adaptive decision-making system designed for collision risk prediction, adaptive warnings, and real-time simulations. The system integrates machine learning with multi-sensor data for real-time analysis of dashcam footage.  

## Features  
- Object detection using YOLOv8  
- Risk level assessment using a trained decision tree model  
- Real-time video processing with simulated sensor data  
- Visual warnings for high-risk objects  

## Installation  

### 1. Clone the Repository  
```sh
git clone NishaadG/IEEE_CodeClash
```

### 2. Set Up a Virtual Environment
```sh
python -m venv risk-assessment-env  
source risk-assessment-env/bin/activate  # macOS/Linux
risk-assessment-env\Scripts\activate  # Windows 
```

### 3. Install Dependencies
```sh
python -m pip install -r requirements.txt
```

## Running the System
### Verify Installation

```sh
python src/utils/verify_installation.py
```

### Simulate Sensor Data

```sh
python src/utils/simulate_sensor_data.py
```

### Train the Risk Assessment Model

```sh
python src/risk_assessment.py  
```

### Run Object Detection & Risk Assessment on Video

```sh
python src/real_time_risk.py  
```
The video can be changed by saving a different video to the videos directory and then altering the path in real_time_risk.py accordingly.