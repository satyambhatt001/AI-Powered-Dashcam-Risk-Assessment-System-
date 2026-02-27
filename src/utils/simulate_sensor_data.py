import numpy as np
import pandas as pd
import os

def generate_sensor_data(num_samples=100):
    np.random.seed(42)
    
    data = pd.DataFrame({
        "object_distance": np.random.uniform(1, 50, num_samples),
        "velocity": np.random.uniform(0, 20, num_samples),
        "trajectory_angle": np.random.uniform(-90, 90, num_samples),
        "vehicle_speed": np.random.uniform(10, 80, num_samples),
        "collision_risk": np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    })
    
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", "sensor_data.csv")
    data.to_csv(file_path, index=False)
    
    print(f"Simulated sensor data saved as '{file_path}'.")

if __name__ == "__main__":
    generate_sensor_data()
