"""
web_server.py -- Flask backend serving both the API and (optionally)
the React UI built at web_ui/dist/.

Start with start_web_server() from the tracker's main(). The server
runs in a daemon thread. It reads/writes the shared `state` dict and
pushes commands into `command_queue`. The tracker main loop drains
that queue and publishes frames into `frame_holder`.
"""

import os
import json
import time
import threading

import cv2
from flask import (Flask, render_template, send_from_directory,
                   Response, jsonify, request)

from shared import command_queue, state, frame_lock, frame_holder

import config


HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "web_ui", "dist")
TEMPLATE_DIR = os.path.join(HERE, "templates")

app = Flask(__name__,
            static_folder=STATIC_DIR,
            static_url_path="/assets",
            template_folder=TEMPLATE_DIR)


# Where enrollment crops and the classifier live
FACE_TRAINING_DIR = "/home/medpal/tracking_person/face_training"
SVM_MODEL_PATH    = "/home/medpal/tracking_person/models/arcface_classifier.pkl"
RFID_PERSONS_FILE = "/home/medpal/tracking_person/rfid_persons.json"
AUTH_UIDS_FILE    = "/home/medpal/tracking_person/authorized_uids.txt"


# ------------------------------------------------------------------
# Motors instance access (set by the tracker at startup)
# ------------------------------------------------------------------
_motors_ref = {"instance": None}


def set_motors_instance(m):
    _motors_ref["instance"] = m


def get_motors():
    return _motors_ref["instance"]


# ------------------------------------------------------------------
# Video streaming
# ------------------------------------------------------------------
def _gen_frames():
    while True:
        with frame_lock:
            frame = frame_holder["frame"]
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------------
# Index page (React build preferred, inline template fallback)
# ------------------------------------------------------------------
@app.route("/")
def index():
    if os.path.isdir(STATIC_DIR):
        return send_from_directory(STATIC_DIR, "index.html")
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# ------------------------------------------------------------------
# Status helpers
# ------------------------------------------------------------------
def _list_enrolled_people():
    if not os.path.isdir(FACE_TRAINING_DIR):
        return []
    return sorted(d for d in os.listdir(FACE_TRAINING_DIR)
                  if os.path.isdir(os.path.join(FACE_TRAINING_DIR, d)))


def _read_wifi_rssi():
    """Best-effort RSSI in dBm from /proc/net/wireless. Falls back to -60."""
    try:
        with open("/proc/net/wireless") as f:
            for line in f.readlines()[2:]:
                parts = line.split()
                if len(parts) >= 4 and parts[0].startswith("wlan"):
                    return int(float(parts[3].rstrip(".")))
    except Exception:
        pass
    return -60


def _read_battery_pct():
    """Placeholder. Replace with a real fuel-gauge read if you have one."""
    return 92


def _normalize_mode():
    """Map internal tracker state to the modes the UI knows."""
    if state.get("enroll_name"):
        return "enrolling"
    m = state.get("mode", "idle")
    if m.startswith("enroll"):
        return "enrolling"
    if m == "training":
        return "training"
    if m == "stopped":
        return "stopped"
    if m == "manual":
        return "manual"
    if state.get("following") and state.get("follow_name"):
        return "following"
    if state.get("servo_open"):
        return "dispensing"
    return "idle"


# ------------------------------------------------------------------
# Status (single route -- this is the ONLY one)
# ------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    dist = state.get("target_distance_mm")
    if dist is None:
        zone = "no_depth"
    elif dist < getattr(config, "REVERSE_DISTANCE_MM", 400):
        zone = "reverse"
    elif dist < getattr(config, "FOLLOW_MIN_MM", 500):
        zone = "hold"
    elif dist < getattr(config, "FOLLOW_MAX_MM", 900):
        zone = "follow"
    else:
        zone = "far"

    return jsonify({
        "mode": _normalize_mode(),
        "follow_name": state.get("follow_name"),
        "servo_open": state.get("servo_open", False),
        "people": _list_enrolled_people(),
        "battery_pct": _read_battery_pct(),
        "wifi_rssi": _read_wifi_rssi(),

        # Follow telemetry
        "zone": zone,
        "target_distance_mm": dist,
        "locked": state.get("person_locked", False),
        "person_locked_name": state.get("person_locked_name"),
        "locked_count": state.get("locked_count", 0),
        "track_count": state.get("track_count", 0),
        "manual_cmd": state.get("manual_cmd"),
        "following": state.get("following", False),
        "motors_available": state.get("motors_available", False),

        # Depth
        "depth_resolution": state.get("depth_resolution"),

        # Enrollment
        "enroll_name": state.get("enroll_name"),
        "enroll_captured": state.get("enroll_captured", 0),
        "enroll_target": state.get("enroll_target", 0),

        # RFID
        "rfid_pending": state.get("rfid_pending"),
    })


# ------------------------------------------------------------------
# Manual motor control
# ------------------------------------------------------------------
@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.json or {}
    cmd = data.get("command")
    if cmd not in ("forward", "backward", "left", "right", "stop"):
        return jsonify({"status": "error",
                        "msg": "unknown command"}), 400
    command_queue.put(("motor", cmd))
    return jsonify({"status": "ok", "command": cmd})


# ------------------------------------------------------------------
# Servo
# ------------------------------------------------------------------
@app.route("/api/servo", methods=["POST"])
def api_servo():
    data = request.json or {}
    action = data.get("action")
    if action not in ("open", "close", "toggle"):
        return jsonify({"status": "error",
                        "msg": "action must be open/close/toggle"}), 400
    command_queue.put(("servo", action))
    return jsonify({"status": "ok", "action": action})


# ------------------------------------------------------------------
# Enrollment + training
# ------------------------------------------------------------------
@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    samples = int(data.get("samples", 30))
    if not name:
        return jsonify({"status": "error", "msg": "name required"}), 400
    command_queue.put(("enroll", {"name": name, "samples": samples}))
    return jsonify({"status": "started", "name": name, "samples": samples})


@app.route("/api/train", methods=["POST"])
def api_train():
    command_queue.put(("train", None))
    return jsonify({"status": "started"})


@app.route("/api/reload_classifier", methods=["POST"])
def api_reload_classifier():
    command_queue.put(("reload_classifier", None))
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------
# Follow target
# ------------------------------------------------------------------
@app.route("/api/follow", methods=["POST"])
def api_follow():
    data = request.json or {}
    name = data.get("name")
    if name is None or (isinstance(name, str) and not name.strip()):
        command_queue.put(("follow", None))
        return jsonify({"status": "ok", "name": None})
    name = name.strip()
    command_queue.put(("follow", name))
    return jsonify({"status": "ok", "name": name})


# ------------------------------------------------------------------
# RFID storage helpers
# ------------------------------------------------------------------
def _load_rfid_persons():
    if not os.path.exists(RFID_PERSONS_FILE):
        return {}
    try:
        with open(RFID_PERSONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_rfid_persons(mapping):
    os.makedirs(os.path.dirname(RFID_PERSONS_FILE), exist_ok=True)
    with open(RFID_PERSONS_FILE, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)


def _load_auth_uids():
    if not os.path.exists(AUTH_UIDS_FILE):
        return set()
    with open(AUTH_UIDS_FILE) as f:
        return {l.strip().upper() for l in f
                if l.strip() and not l.startswith("#")}


def _save_auth_uids(uids):
    os.makedirs(os.path.dirname(AUTH_UIDS_FILE), exist_ok=True)
    with open(AUTH_UIDS_FILE, "w") as f:
        f.write("# Authorized RFID UIDs (hex uppercase)\n")
        for u in sorted(uids):
            f.write(u + "\n")


# ------------------------------------------------------------------
# RFID routes
# ------------------------------------------------------------------
@app.route("/api/rfid/list")
def api_rfid_list():
    mapping = _load_rfid_persons()
    uids = _load_auth_uids()
    items = [{"uid": u, "person": mapping.get(u),
              "authorized": u in uids}
             for u in sorted(set(mapping) | uids)]
    return jsonify({"items": items,
                    "pending": state.get("rfid_pending")})


@app.route("/api/rfid/enroll/start", methods=["POST"])
def api_rfid_enroll_start():
    data = request.json or {}
    person = (data.get("person") or "").strip()
    if not person:
        return jsonify({"status": "error",
                        "msg": "person required"}), 400
    motors = get_motors()
    if motors is None or not motors.available:
        return jsonify({"status": "error",
                        "msg": "ESP32 not connected"}), 503
    if not motors.rfid_enroll(person):
        return jsonify({"status": "error",
                        "msg": "invalid person name"}), 400
    state["rfid_pending"] = person
    return jsonify({"status": "ok", "person": person, "timeout_s": 20})


@app.route("/api/rfid/enroll/cancel", methods=["POST"])
def api_rfid_enroll_cancel():
    motors = get_motors()
    if motors is not None:
        motors.rfid_enroll_cancel()
    state["rfid_pending"] = None
    return jsonify({"status": "ok"})


@app.route("/api/rfid/remove", methods=["POST"])
def api_rfid_remove():
    data = request.json or {}
    uid = (data.get("uid") or "").strip().upper()
    if not uid:
        return jsonify({"status": "error", "msg": "uid required"}), 400
    mapping = _load_rfid_persons()
    mapping.pop(uid, None)
    _save_rfid_persons(mapping)
    uids = _load_auth_uids()
    uids.discard(uid)
    _save_auth_uids(uids)
    return jsonify({"status": "ok", "removed": uid})


# ------------------------------------------------------------------
# RFID event callbacks (called from SerialMotors reader thread)
# ------------------------------------------------------------------
def on_rfid_enrolled(person, uid):
    mapping = _load_rfid_persons()
    mapping[uid] = person
    _save_rfid_persons(mapping)
    uids = _load_auth_uids()
    uids.add(uid)
    _save_auth_uids(uids)
    state["rfid_pending"] = None
    print(f"[rfid] enrolled {uid} -> {person}")


def on_rfid_timeout():
    state["rfid_pending"] = None
    print("[rfid] enrollment timeout")


# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
def start_web_server(host="0.0.0.0", port=5000):
    if not os.path.isdir(STATIC_DIR):
        print(f"[web] WARNING: {STATIC_DIR} not found. "
              f"Falling back to templates/index.html")

    def _run():
        app.run(host=host, port=port,
                debug=False, threaded=True, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"[web] control panel at http://{host}:{port}")
    return t
