import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SensorData:
    sensor_id: int
    timestamp: float
    temperature: float
    humidity: float

class Sensor:
    def __init__(self, sensor_id: int, start_temp: float = 20.0, start_hum: float = 50.0):
        self.sensor_id = sensor_id
        self.temp = start_temp
        self.hum = start_hum
        # Random walk parameters
        self.temp_noise_std = 0.5
        self.hum_noise_std = 1.0

    def read(self) -> SensorData:
        # Random walk
        self.temp += np.random.normal(0, self.temp_noise_std)
        self.hum += np.random.normal(0, self.hum_noise_std)
        
        # Clip values to realistic ranges
        self.temp = np.clip(self.temp, -10, 50)
        self.hum = np.clip(self.hum, 0, 100)
        
        return SensorData(
            sensor_id=self.sensor_id,
            timestamp=time.time(),
            temperature=self.temp,
            humidity=self.hum
        )

class Network:
    def __init__(self, num_sensors: int):
        self.sensors = [Sensor(i) for i in range(num_sensors)]

    def collect_data(self) -> List[SensorData]:
        return [sensor.read() for sensor in self.sensors]

    def generate_dataset(self, steps: int) -> pd.DataFrame:
        """Generates a dataset for a given number of steps."""
        data = []
        for _ in range(steps):
            data.extend([s.read() for s in self.sensors])
        
        df = pd.DataFrame([vars(d) for d in data])
        return df
