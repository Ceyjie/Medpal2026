# shared.py  (import from both files)
import threading
import queue

# 1. Commands: web / touch / RFID → tracker main loop
command_queue = queue.Queue()

# 2. State: tracker → web (status, servo, follow target, enrollment)
state = {
    "following": False,
    "follow_name": None,
    "servo_open": False,
    "mode": "idle",
    "last_error": None,
    "motors_available": False,
}

# 3. Frame: tracker → web (latest JPEG-encodable BGR frame)
frame_lock = threading.Lock()
frame_holder = {"frame": None}