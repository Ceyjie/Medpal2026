"""
SerialMotors -- drop-in replacement for tracker.py's Motors class.

Same public interface (set_speed, stop, forward, backward, turn_left,
turn_right, .available), but sends commands to an ESP32 over USB serial
instead of driving Pi GPIO directly. To switch tracker.py over:

    # was:
    motors = Motors()
    # becomes:
    from serial_motors import SerialMotors
    motors = SerialMotors()

Nothing else in tracker.py's motor-control logic needs to change.
"""
import time

try:
    import serial
except ImportError:
    serial = None


class SerialMotors:
    def __init__(self, port=None, baud=115200):
        self.available = False
        self.current_speed = 40  # percent, matches config.FOLLOW_BASE_SPEED-ish
        self.ser = None

        if serial is None:
            print("Motors unavailable: pyserial not installed (pip install pyserial)")
            return

        port = port or self._autodetect_port()
        if port is None:
            print("Motors unavailable: no ESP32 serial port found")
            return

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # ESP32 resets on serial open, needs a moment to boot
            self.ser.reset_input_buffer()
            self._send("PING")
            reply = self.ser.readline().decode(errors="ignore").strip()
            if reply != "PONG":
                print(f"Motors unavailable: no PONG reply from ESP32 (got: {reply!r})")
                return
            self.available = True
            print(f"Motors initialized (serial, {port})")
        except Exception as e:
            print(f"Motors unavailable: {e}")

    @staticmethod
    def _autodetect_port():
        import glob
        candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        return candidates[0] if candidates else None

    def _send(self, line):
        if self.ser is None:
            return
        self.ser.write((line + "\n").encode())

    def set_speed(self, speed):
        self.current_speed = max(0, min(100, speed))
        print(f"Speed: {self.current_speed}%")
        return self.current_speed / 100

    def stop(self):
        if not self.available:
            return
        self._send("S")

    def forward(self):
        if not self.available:
            return
        self._send(f"F:{self.current_speed}")

    def backward(self):
        if not self.available:
            return
        self._send(f"B:{self.current_speed}")

    def turn_left(self):
        if not self.available:
            return
        self._send(f"L:{self.current_speed}")

    def turn_right(self):
        if not self.available:
            return
        self._send(f"R:{self.current_speed}")
