"""
web_server.py -- Flask control panel that runs alongside the tracker.

Start with start_web_server() from the tracker's main(). The server
runs in a daemon thread. It reads/writes the shared `state` dict and
pushes commands into `command_queue`. The tracker main loop drains
that queue and publishes frames into `frame_holder`.
"""

import os
import json
import time
import threading
import subprocess

import cv2
from flask import (Flask, render_template, Response,
                   jsonify, request)

from shared import command_queue, state, frame_lock, frame_holder


app = Flask(__name__)

# Where enrollment crops and the classifier live
FACE_TRAINING_DIR = "/home/medpal/tracking_person/face_training"
SVM_MODEL_PATH    = "/home/medpal/tracking_person/models/arcface_classifier.pkl"
RFID_PERSONS_FILE = "/home/medpal/tracking_person/rfid_persons.json"
AUTH_UIDS_FILE    = "/home/medpal/tracking_person/authorized_uids.txt"

# Set by the tracker at startup
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
        time.sleep(0.05)   # ~20 FPS cap for streaming


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; "
                             "boundary=frame")


# ------------------------------------------------------------------
# Index page
# ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    with frame_lock:
        pass
    return jsonify({
        "following": state["following"],
        "follow_name": state["follow_name"],
        "servo_open": state["servo_open"],
        "mode": state["mode"],
        "motors_available": state["motors_available"],
        "people": _list_enrolled_people(),
    })


def _list_enrolled_people():
    if not os.path.isdir(FACE_TRAINING_DIR):
        return []
    return sorted(d for d in os.listdir(FACE_TRAINING_DIR)
                  if os.path.isdir(os.path.join(FACE_TRAINING_DIR, d)))


# ------------------------------------------------------------------
# Manual motor control
# ------------------------------------------------------------------
@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.json or {}
    cmd = data.get("command")
    if cmd not in ("forward", "backward", "left",
                   "right", "stop"):
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
        return jsonify({"status": "error",
                        "msg": "name required"}), 400
    command_queue.put(("enroll", {"name": name, "samples": samples}))
    return jsonify({"status": "started", "name": name,
                    "samples": samples})


@app.route("/api/train", methods=["POST"])
def api_train():
    command_queue.put(("train", None))
    return jsonify({"status": "started"})


# ------------------------------------------------------------------
# Follow target
# ------------------------------------------------------------------
@app.route("/api/follow", methods=["POST"])
def api_follow():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error",
                        "msg": "name required"}), 400
    command_queue.put(("follow", name))
    return jsonify({"status": "ok", "name": name})


# ------------------------------------------------------------------
# RFID management
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


@app.route("/api/rfid/list")
def api_rfid_list():
    mapping = _load_rfid_persons()
    uids = _load_auth_uids()
    items = [{"uid": u, "person": mapping.get(u),
              "authorized": u in uids}
             for u in sorted(set(mapping) | uids)]
    return jsonify({"items": items})


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
    return jsonify({"status": "ok", "person": person, "timeout_s": 20})


@app.route("/api/rfid/enroll/cancel", methods=["POST"])
def api_rfid_enroll_cancel():
    motors = get_motors()
    if motors is not None:
        motors.rfid_enroll_cancel()
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
    print(f"[rfid] enrolled {uid} -> {person}")


def on_rfid_timeout():
    print("[rfid] enrollment timeout")




@app.route("/api/reload_classifier", methods=["POST"])
def api_reload_classifier():
    command_queue.put(("reload_classifier", None))
    return jsonify({"status": "ok"})




# ------------------------------------------------------------------
# Start
# ------------------------------------------------------------------
def start_web_server(host="0.0.0.0", port=5000):
    """Launch Flask in a daemon thread. Returns the thread."""
    def _run():
        app.run(host=host, port=port,
                debug=False,
                threaded=True,
                use_reloader=False)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"[web] http://{host}:{port}  (threaded=True)")
    return t
