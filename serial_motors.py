"""
serial_motors.py -- Pi-side driver for the MedPal ESP32.

Motors, servo, touch, RFID, and RFID enrollment.
"""

import time
import threading

try:
    import serial
except ImportError:
    serial = None


class SerialMotors:
    def __init__(self, port=None, baud=115200,
                 on_touch=None,
                 on_rfid=None,
                 on_rfid_enrolled=None,
                 on_rfid_timeout=None):
        """
        Args:
            on_touch:          callable(pressed: bool)
            on_rfid:           callable(uid: str)         -- normal scan
            on_rfid_enrolled:  callable(person, uid)      -- enroll scan
            on_rfid_timeout:   callable()                 -- enroll timed out
        """
        self.available = False
        self.current_speed = 40
        self.ser = None
        self.on_touch = on_touch
        self.on_rfid = on_rfid
        self.on_rfid_enrolled = on_rfid_enrolled
        self.on_rfid_timeout = on_rfid_timeout
        self._reader_thread = None
        self._running = False

        if serial is None:
            print("Motors unavailable: pyserial not installed")
            return

        port = port or self._autodetect_port()
        if port is None:
            print("Motors unavailable: no ESP32 serial port found")
            return

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            self._send("PING")
            reply = self.ser.readline().decode(errors="ignore").strip()
            if reply != "PONG":
                print(f"Motors unavailable: no PONG (got {reply!r})")
                return
            self.available = True
            print(f"Motors initialized (serial, {port})")

            self._running = True
            self._reader_thread = threading.Thread(
                target=self._read_loop, daemon=True)
            self._reader_thread.start()
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
        try:
            self.ser.write((line + "\n").encode())
        except Exception as e:
            print(f"serial write failed: {e}")

    # ---------------------------------------------------------------
    # Background reader
    # ---------------------------------------------------------------
    def _read_loop(self):
        while self._running:
            try:
                line = self.ser.readline().decode(
                    errors="ignore").strip()
                if not line:
                    continue
                self._dispatch_event(line)
            except Exception:
                time.sleep(0.05)

    def _dispatch_event(self, line):
        # Ignore routine acknowledgments
        if line in ("PONG", "OK STOP"):
            return
        if line.startswith("OK "):
            # Log enrollment acknowledgments; ignore others silently
            print(f"[esp32] {line}")
            return

        # Enrollment success: RFID_ENROLLED:<person>:<uid>
        if line.startswith("RFID_ENROLLED:"):
            rest = line[len("RFID_ENROLLED:"):]
            if ":" in rest:
                person, uid = rest.split(":", 1)
                if self.on_rfid_enrolled:
                    try:
                        self.on_rfid_enrolled(person.strip(),
                                              uid.strip().upper())
                    except Exception as e:
                        print(f"on_rfid_enrolled error: {e}")
            return

        # Enrollment timeout
        if line == "RFID_ENROLL_TIMEOUT":
            if self.on_rfid_timeout:
                try:
                    self.on_rfid_timeout()
                except Exception as e:
                    print(f"on_rfid_timeout error: {e}")
            return

        # Normal scan: RFID:<uid>
        if line.startswith("RFID:"):
            uid = line.split(":", 1)[1].strip().upper()
            if self.on_rfid:
                try:
                    self.on_rfid(uid)
                except Exception as e:
                    print(f"on_rfid handler error: {e}")
            return

        # Touch
        if line.startswith("TOUCH:"):
            val = line.split(":", 1)[1].strip()
            pressed = (val == "1")
            if self.on_touch:
                try:
                    self.on_touch(pressed)
                except Exception as e:
                    print(f"on_touch handler error: {e}")
            return

        if line.startswith("ERR"):
            print(f"[esp32] {line}")
            return

        print(f"[esp32] {line}")

    # ---------------------------------------------------------------
    # Motor commands
    # ---------------------------------------------------------------
    def set_speed(self, speed):
        new_speed = max(0, min(100, speed))
        self.current_speed = new_speed
        return self.current_speed / 100

    def stop(self):
        if self.available: self._send("S")

    def forward(self):
        if self.available: self._send(f"F:{self.current_speed}")

    def backward(self):
        if self.available: self._send(f"B:{self.current_speed}")

    def turn_left(self):
        if self.available: self._send(f"L:{self.current_speed}")

    def turn_right(self):
        if self.available: self._send(f"R:{self.current_speed}")

    # ---------------------------------------------------------------
    # Servo commands
    # ---------------------------------------------------------------
    def servo_open(self):
        if self.available: self._send("SV:OPEN")

    def servo_close(self):
        if self.available: self._send("SV:CLOSE")

    def servo_angle(self, deg):
        deg = max(0, min(180, int(deg)))
        if self.available: self._send(f"SV:{deg}")

    # ---------------------------------------------------------------
    # RFID enrollment
    # ---------------------------------------------------------------
    def rfid_enroll(self, person):
        """Enter enrollment mode for `person`. The next card scanned
        will fire `on_rfid_enrolled(person, uid)`."""
        if not self.available:
            return False
        person = person.strip()
        if not person or ":" in person or "\n" in person:
            print(f"rfid_enroll: invalid person {person!r}")
            return False
        self._send(f"RFID_ENROLL:{person}")
        return True

    def rfid_enroll_cancel(self):
        if self.available:
            self._send("RFID_ENROLL_CANCEL")

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------
    def cleanup(self):
        self._running = False
        try:
            if self.available:
                self._send("S")
                self._send("SV:CLOSE")
        except Exception:
            pass
        if self._reader_thread is not None:
            try:
                self._reader_thread.join(timeout=0.5)
            except Exception:
                pass
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

    def test_motors(self):
        if not self.available:
            print("Motors unavailable")
            return
        print("Testing motors...")
        self.set_speed(40)
        for action in [self.forward, self.backward,
                       self.turn_left, self.turn_right]:
            action(); time.sleep(0.5)
            self.stop(); time.sleep(0.2)
        print("Motor test complete.")
