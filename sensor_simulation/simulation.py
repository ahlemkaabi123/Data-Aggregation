import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from network_topology import NetworkNode, Packet
from processing import detect_anomalies

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

from network_topology import NetworkNode, Packet
from processing import detect_anomalies

class Network:
    def __init__(self, num_sensors: int):
        self.num_sensors = num_sensors
        self.sensors = [Sensor(i) for i in range(num_sensors)]
        self.nodes = [NetworkNode(i, 'sensor') for i in range(num_sensors)]
        
        # Setup Topology (Simple Tree)
        # Node 0 is Sink
        self.nodes[0].node_type = 'sink'
        self.sink_id = 0
        
        # Connect others to random parent with lower ID (to ensure tree rooted at 0)
        # This is a simple way to create a connected DAG/Tree
        for i in range(1, num_sensors):
            parent = np.random.randint(0, i)
            self.nodes[i].set_parent(parent)
            self.nodes[i].neighbors.append(parent)
            # Parent also knows child (neighbor)
            self.nodes[parent].neighbors.append(i)

    def generate_dataset(self, steps: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates a dataset by running the distributed simulation.
        Returns: (raw_df, sink_df)
        """
        raw_data = []
        sink_data = []
        
        for step in range(steps):
            # 1. Sensors Read Data
            step_readings = []
            for sensor in self.sensors:
                reading = sensor.read()
                step_readings.append(reading)
                
                # 2. Nodes generate packets
                # In this simulation, we assume 1-to-1 mapping of Sensor to NetworkNode
                node = self.nodes[sensor.sensor_id]
                packet = node.generate_data_packet(vars(reading))
                
                # Simple Anomaly Detection (Online)
                # We need history for Z-score. Let's cheat a bit and look at previous raw_data for this sensor
                # Or just use a simple threshold for now since we are generating random walk
                if reading.temperature > 40 or reading.temperature < 0: # Simple threshold
                     packet.type = 'anomaly'
                
                node.receive_packet(packet)
            
            raw_data.extend(step_readings)
            
            # 3. Network Transmission (Multi-hop)
            # We simulate "rounds" of transmission to propagate data to sink
            # In a real network, this happens asynchronously.
            # Here, we iterate from leaves to root (reverse order of IDs) to move data up
            
            # We do multiple sub-steps to allow multi-hop in one simulation step
            # or just one hop per step?
            # User asked to "Simulate transmission delays".
            # Let's do one hop per simulation step for simplicity, or a few hops.
            # To ensure data reaches sink, we can iterate enough times.
            
            max_hops = self.num_sensors
            for _ in range(max_hops):
                for i in range(self.num_sensors - 1, 0, -1): # Skip sink (0)
                    node = self.nodes[i]
                    outgoing = node.process_buffer()
                    if outgoing and node.parent_id is not None:
                        parent = self.nodes[node.parent_id]
                        for pkt in outgoing:
                            parent.receive_packet(pkt)
                            
            # 4. Sink Processing
            sink = self.nodes[self.sink_id]
            # Sink "receives" everything in its buffer
            # We just move it to sink_data
            while sink.buffer:
                pkt = sink.buffer.pop(0)
                # Add metadata about path
                record = pkt.data.copy()
                record['path'] = pkt.path
                record['type'] = pkt.type
                sink_data.append(record)
                
        raw_df = pd.DataFrame([vars(d) for d in raw_data])
        sink_df = pd.DataFrame(sink_data)
        
        return raw_df, sink_df
