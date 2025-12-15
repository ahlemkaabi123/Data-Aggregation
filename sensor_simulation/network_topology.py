import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class Packet:
    source_id: int
    data: Any
    timestamp: float
    path: List[int]
    type: str = 'data' # 'data', 'anomaly'

class NetworkNode:
    """
    Represents a node in the distributed sensor network.
    Can be a Sensor, Relay, or Sink.
    """
    def __init__(self, node_id: int, node_type: str = 'sensor', neighbors: List[int] = None):
        self.node_id = node_id
        self.node_type = node_type # 'sensor', 'relay', 'sink'
        self.neighbors = neighbors if neighbors else []
        self.buffer: List[Packet] = []
        self.parent_id: Optional[int] = None # For tree topology
        
        # Simulation state
        self.battery = 100.0
        
    def set_parent(self, parent_id: int):
        self.parent_id = parent_id
        
    def receive_packet(self, packet: Packet):
        """Receives a packet from a neighbor."""
        # Simulate processing delay
        # time.sleep(0.001) 
        self.buffer.append(packet)
        
    def process_buffer(self) -> List[Packet]:
        """
        Processes packets in the buffer.
        Aggregates data if this is a relay node.
        Returns a list of packets to send to the next hop.
        """
        if not self.buffer:
            return []
            
        outgoing_packets = []
        
        # Separate anomalies from regular data
        anomalies = [p for p in self.buffer if p.type == 'anomaly']
        regular_data = [p for p in self.buffer if p.type == 'data']
        
        # Forward anomalies immediately without aggregation
        for anomaly in anomalies:
            anomaly.path.append(self.node_id)
            outgoing_packets.append(anomaly)
            
        # Aggregate regular data if we have enough or if it's time
        if regular_data:
            # For now, just forward (aggregation logic will be added later)
            # In a real distributed system, we would wait for data from all children
            # or a timeout. Here we simplify.
            
            # If I am a relay, I might want to aggregate my own data with received data
            # But for now, let's just assume we forward everything to parent
            # The actual aggregation logic will be called by the simulation controller
            # or we can implement a simple aggregation here.
            
            # Let's just forward for now to establish connectivity
            for packet in regular_data:
                packet.path.append(self.node_id)
                outgoing_packets.append(packet)
                
        self.buffer = [] # Clear buffer
        return outgoing_packets

    def generate_data_packet(self, sensor_read: Dict) -> Packet:
        """Creates a packet from local sensor reading."""
        return Packet(
            source_id=self.node_id,
            data=sensor_read,
            timestamp=time.time(),
            path=[self.node_id],
            type='data'
        )
