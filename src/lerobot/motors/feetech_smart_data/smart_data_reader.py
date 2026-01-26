"""
Feetech STS3215 Smart Data Reader

Provides efficient reading of diagnostic/telemetry data from Feetech STS3215 motors.
Supports both individual motor reads and batch sync_read for multiple motors.
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

# STS3215 Smart Data Register Definitions
# Format: name -> (address, size_bytes, description, units, conversion_func)
SMART_DATA_REGISTERS = {
    "Present_Position": {
        "address": 56,
        "size": 2,
        "description": "Current position",
        "units": "encoder_counts",
        "range": (0, 4095),
    },
    "Present_Velocity": {
        "address": 58,
        "size": 2,
        "description": "Current velocity",
        "units": "steps_per_sec",
        "signed": True,  # Sign-magnitude encoding, bit 15 is sign
        "sign_bit": 15,
    },
    "Present_Load": {
        "address": 60,
        "size": 2,
        "description": "Current load/torque",
        "units": "percent_x10",  # 0.1% units
        "signed": True,  # Sign-magnitude encoding, bit 10 is sign
        "sign_bit": 10,
    },
    "Present_Voltage": {
        "address": 62,
        "size": 1,
        "description": "Input voltage",
        "units": "decivolts",  # Value / 10 = volts
        "scale": 0.1,
    },
    "Present_Temperature": {
        "address": 63,
        "size": 1,
        "description": "Motor temperature",
        "units": "celsius",
    },
    "Status": {
        "address": 65,
        "size": 1,
        "description": "Error status flags",
        "units": "bitmask",
    },
    "Moving": {
        "address": 66,
        "size": 1,
        "description": "Motor in motion flag",
        "units": "bool",
    },
    "Present_Current": {
        "address": 69,
        "size": 2,
        "description": "Motor current draw",
        "units": "mA",  # milliamps (6.5mA per unit based on datasheet)
        "scale": 6.5,
    },
}

# Status flag bit definitions
STATUS_FLAGS = {
    0: "voltage_error",      # Voltage out of range
    1: "angle_limit_error",  # Position limit exceeded
    2: "overheat_error",     # Temperature too high
    3: "overload_error",     # Load exceeded limit
    4: "overcurrent_error",  # Current exceeded limit
}


@dataclass
class MotorSmartData:
    """Container for smart data from a single motor."""
    motor_id: int
    timestamp: float = field(default_factory=time.time)
    
    # Raw values
    position_raw: Optional[int] = None
    velocity_raw: Optional[int] = None
    load_raw: Optional[int] = None
    voltage_raw: Optional[int] = None
    temperature_raw: Optional[int] = None
    status_raw: Optional[int] = None
    moving_raw: Optional[int] = None
    current_raw: Optional[int] = None
    
    @property
    def position(self) -> Optional[int]:
        """Position in encoder counts (0-4095)."""
        return self.position_raw
    
    @property
    def velocity(self) -> Optional[int]:
        """Velocity in steps/sec (signed)."""
        if self.velocity_raw is None:
            return None
        return self._decode_signed(self.velocity_raw, 15)
    
    @property
    def load(self) -> Optional[float]:
        """Load as percentage (-100% to +100%)."""
        if self.load_raw is None:
            return None
        signed_val = self._decode_signed(self.load_raw, 10)
        return signed_val / 10.0  # Convert from 0.1% units
    
    @property
    def voltage(self) -> Optional[float]:
        """Voltage in volts."""
        if self.voltage_raw is None:
            return None
        return self.voltage_raw / 10.0
    
    @property
    def temperature(self) -> Optional[int]:
        """Temperature in Celsius."""
        return self.temperature_raw
    
    @property
    def current(self) -> Optional[float]:
        """Current in milliamps."""
        if self.current_raw is None:
            return None
        return self.current_raw * 6.5  # 6.5mA per unit
    
    @property
    def moving(self) -> Optional[bool]:
        """True if motor is in motion."""
        if self.moving_raw is None:
            return None
        return bool(self.moving_raw)
    
    @property
    def status(self) -> Optional[int]:
        """Raw status byte."""
        return self.status_raw
    
    @property
    def status_flags(self) -> dict[str, bool]:
        """Decoded status flags."""
        if self.status_raw is None:
            return {}
        return {
            name: bool(self.status_raw & (1 << bit))
            for bit, name in STATUS_FLAGS.items()
        }
    
    @property
    def has_error(self) -> bool:
        """True if any error flag is set."""
        return self.status_raw is not None and self.status_raw != 0
    
    @staticmethod
    def _decode_signed(value: int, sign_bit: int) -> int:
        """Decode sign-magnitude encoded value."""
        mask = (1 << sign_bit) - 1
        magnitude = value & mask
        if value & (1 << sign_bit):
            return -magnitude
        return magnitude
    
    def to_dict(self) -> dict:
        """Convert to dictionary with all processed values."""
        return {
            "motor_id": self.motor_id,
            "timestamp": self.timestamp,
            "position": self.position,
            "velocity": self.velocity,
            "load_percent": self.load,
            "voltage_v": self.voltage,
            "temperature_c": self.temperature,
            "current_ma": self.current,
            "moving": self.moving,
            "status": self.status,
            "status_flags": self.status_flags,
            "has_error": self.has_error,
        }
    
    def __repr__(self) -> str:
        return (
            f"MotorSmartData(id={self.motor_id}, "
            f"pos={self.position}, vel={self.velocity}, "
            f"load={self.load:.1f}%, temp={self.temperature}C, "
            f"V={self.voltage:.1f}V, I={self.current:.1f}mA)"
        )


@dataclass
class SmartDataBatch:
    """Container for smart data from multiple motors."""
    timestamp: float = field(default_factory=time.time)
    motors: dict[int, MotorSmartData] = field(default_factory=dict)
    
    def __getitem__(self, motor_id: int) -> MotorSmartData:
        return self.motors[motor_id]
    
    def __iter__(self):
        return iter(self.motors.values())
    
    def __len__(self):
        return len(self.motors)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "motors": {mid: m.to_dict() for mid, m in self.motors.items()},
        }


class FeetechSmartDataReader:
    """
    Reader for Feetech STS3215 motor smart data.
    
    Can be used standalone with a port, or integrated with an existing
    lerobot MotorsBus instance.
    
    Example standalone usage:
        reader = FeetechSmartDataReader("/dev/manipulator_left")
        reader.connect()
        data = reader.read_all([1, 2, 3, 4, 5, 6])
        print(data)
        reader.disconnect()
    
    Example with existing bus:
        # If you have a FeetechMotorsBus already connected
        reader = FeetechSmartDataReader.from_bus(existing_bus)
        data = reader.read_all()
    """
    
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 1_000_000,
        protocol_version: int = 0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        
        self._port_handler = None
        self._packet_handler = None
        self._owns_port = False
        self._connected = False
        self._motor_ids: list[int] = []
    
    @classmethod
    def from_bus(cls, bus) -> "FeetechSmartDataReader":
        """Create a reader that uses an existing FeetechMotorsBus connection."""
        reader = cls()
        reader._port_handler = bus.port_handler
        reader._packet_handler = bus.packet_handler
        reader._owns_port = False
        reader._connected = True
        reader._motor_ids = list(bus.ids)
        return reader
    
    def connect(self) -> bool:
        """Connect to the serial port."""
        if self._connected:
            return True
        
        if self.port is None:
            raise ValueError("Port must be specified for standalone connection")
        
        try:
            import scservo_sdk as scs
            
            self._port_handler = scs.PortHandler(self.port)
            if not self._port_handler.openPort():
                logger.error(f"Failed to open port {self.port}")
                return False
            
            if not self._port_handler.setBaudRate(self.baudrate):
                logger.error(f"Failed to set baudrate {self.baudrate}")
                self._port_handler.closePort()
                return False
            
            self._packet_handler = scs.PacketHandler(self.protocol_version)
            self._owns_port = True
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the serial port (only if we own it)."""
        if self._owns_port and self._port_handler is not None:
            self._port_handler.closePort()
        self._connected = False
    
    def scan(self, id_range: range = range(1, 10)) -> list[int]:
        """Scan for motors on the bus."""
        import scservo_sdk as scs
        
        if not self._connected:
            raise RuntimeError("Not connected")
        
        found = []
        for motor_id in id_range:
            _, result, _ = self._packet_handler.ping(
                self._port_handler, motor_id
            )
            if result == scs.COMM_SUCCESS:
                found.append(motor_id)
        
        self._motor_ids = found
        return found
    
    def read_single(self, motor_id: int) -> MotorSmartData:
        """Read all smart data from a single motor."""
        import scservo_sdk as scs
        
        if not self._connected:
            raise RuntimeError("Not connected")
        
        data = MotorSmartData(motor_id=motor_id)
        
        # Read each register
        for name, reg_info in SMART_DATA_REGISTERS.items():
            addr = reg_info["address"]
            size = reg_info["size"]
            
            if size == 1:
                value, result, _ = self._packet_handler.read1ByteTxRx(
                    self._port_handler, motor_id, addr
                )
            elif size == 2:
                value, result, _ = self._packet_handler.read2ByteTxRx(
                    self._port_handler, motor_id, addr
                )
            else:
                continue
            
            if result == scs.COMM_SUCCESS:
                # Map to dataclass field
                field_name = name.lower() + "_raw"
                if hasattr(data, field_name):
                    setattr(data, field_name, value)
        
        return data
    
    def read_all(
        self,
        motor_ids: Optional[list[int]] = None,
        use_sync_read: bool = True,
    ) -> SmartDataBatch:
        """
        Read smart data from multiple motors.
        
        Args:
            motor_ids: List of motor IDs to read. If None, uses previously scanned IDs.
            use_sync_read: If True, use sync_read for efficiency (requires protocol 0).
        
        Returns:
            SmartDataBatch containing data for all motors.
        """
        if not self._connected:
            raise RuntimeError("Not connected")
        
        ids = motor_ids or self._motor_ids
        if not ids:
            raise ValueError("No motor IDs specified and none discovered via scan()")
        
        batch = SmartDataBatch()
        
        if use_sync_read and self.protocol_version == 0:
            batch = self._sync_read_all(ids)
        else:
            # Fall back to sequential reads
            for motor_id in ids:
                batch.motors[motor_id] = self.read_single(motor_id)
        
        return batch
    
    def _sync_read_all(self, motor_ids: list[int]) -> SmartDataBatch:
        """Use sync_read to efficiently read from multiple motors."""
        import scservo_sdk as scs
        
        batch = SmartDataBatch()
        
        # Initialize motor data
        for mid in motor_ids:
            batch.motors[mid] = MotorSmartData(motor_id=mid, timestamp=batch.timestamp)
        
        # Group registers by contiguous address ranges for efficiency
        # Read position through temperature in one block (addr 56-63, 8 bytes)
        # Then read current separately (addr 69, 2 bytes)
        
        # Block 1: Position, Velocity, Load, Voltage, Temperature (56-63)
        block1_addr = 56
        block1_len = 8  # 56-63
        
        sync_read = scs.GroupSyncRead(
            self._port_handler, self._packet_handler, block1_addr, block1_len
        )
        
        for mid in motor_ids:
            sync_read.addParam(mid)
        
        result = sync_read.txRxPacket()
        if result == scs.COMM_SUCCESS:
            for mid in motor_ids:
                if sync_read.isAvailable(mid, block1_addr, block1_len):
                    # Extract values from the block
                    pos = sync_read.getData(mid, 56, 2)
                    vel = sync_read.getData(mid, 58, 2)
                    load = sync_read.getData(mid, 60, 2)
                    volt = sync_read.getData(mid, 62, 1)
                    temp = sync_read.getData(mid, 63, 1)
                    
                    batch.motors[mid].position_raw = pos
                    batch.motors[mid].velocity_raw = vel
                    batch.motors[mid].load_raw = load
                    batch.motors[mid].voltage_raw = volt
                    batch.motors[mid].temperature_raw = temp
        
        sync_read.clearParam()
        
        # Block 2: Status and Moving (65-66, 2 bytes)
        block2_addr = 65
        block2_len = 2
        
        sync_read2 = scs.GroupSyncRead(
            self._port_handler, self._packet_handler, block2_addr, block2_len
        )
        
        for mid in motor_ids:
            sync_read2.addParam(mid)
        
        result = sync_read2.txRxPacket()
        if result == scs.COMM_SUCCESS:
            for mid in motor_ids:
                if sync_read2.isAvailable(mid, block2_addr, block2_len):
                    status = sync_read2.getData(mid, 65, 1)
                    moving = sync_read2.getData(mid, 66, 1)
                    batch.motors[mid].status_raw = status
                    batch.motors[mid].moving_raw = moving
        
        sync_read2.clearParam()
        
        # Block 3: Current (69, 2 bytes)
        block3_addr = 69
        block3_len = 2
        
        sync_read3 = scs.GroupSyncRead(
            self._port_handler, self._packet_handler, block3_addr, block3_len
        )
        
        for mid in motor_ids:
            sync_read3.addParam(mid)
        
        result = sync_read3.txRxPacket()
        if result == scs.COMM_SUCCESS:
            for mid in motor_ids:
                if sync_read3.isAvailable(mid, block3_addr, block3_len):
                    current = sync_read3.getData(mid, 69, 2)
                    batch.motors[mid].current_raw = current
        
        sync_read3.clearParam()
        
        return batch
    
    def read_continuous(
        self,
        motor_ids: Optional[list[int]] = None,
        interval_ms: float = 100,
        callback=None,
    ):
        """
        Generator that continuously reads smart data.
        
        Args:
            motor_ids: List of motor IDs to read.
            interval_ms: Minimum interval between reads in milliseconds.
            callback: Optional callback function called with each batch.
        
        Yields:
            SmartDataBatch for each read cycle.
        """
        ids = motor_ids or self._motor_ids
        interval_sec = interval_ms / 1000.0
        
        while True:
            start = time.time()
            batch = self.read_all(ids)
            
            if callback:
                callback(batch)
            
            yield batch
            
            # Maintain timing
            elapsed = time.time() - start
            sleep_time = interval_sec - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# Convenience function for quick testing
def read_motors_smart_data(
    port: str,
    motor_ids: Optional[list[int]] = None,
    baudrate: int = 1_000_000,
) -> SmartDataBatch:
    """
    Convenience function to quickly read smart data from motors.
    
    Args:
        port: Serial port path (e.g., "/dev/manipulator_left")
        motor_ids: List of motor IDs. If None, will scan for motors.
        baudrate: Serial baudrate (default 1M for STS3215)
    
    Returns:
        SmartDataBatch with motor data.
    """
    reader = FeetechSmartDataReader(port, baudrate)
    reader.connect()
    
    try:
        if motor_ids is None:
            motor_ids = reader.scan()
        return reader.read_all(motor_ids)
    finally:
        reader.disconnect()
