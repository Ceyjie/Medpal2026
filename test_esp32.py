#!/usr/bin/env python3
"""
PHASE 0 TEST -- ESP32 SERIAL LINK, THEN MOTORS

Two stages, run separately:

  STAGE 1 (do this FIRST, before wiring any motor to the ESP32):
    Flash esp32_motor_controller.ino, connect ESP32 to Pi via USB, run
    this script with no motors wired at all. It just checks PING/PONG
    over serial -- proves the Pi<->ESP32 link itself works before
    anything can spin.

  STAGE 2 (after wiring BTS7960 to the ESP32 per the .ino comments):
    Same script, now also exercises forward/backward/turn via
    SerialMotors -- same menu style as test_motors.py, so results are
    directly comparable to the direct-GPIO version if you ever switch
    back.

What "pass" looks like for Stage 1:
  - "[esp32] Serial link OK -- got PONG" prints, no timeout/no error.
  If this fails: check the port (ls /dev/ttyUSB* /dev/ttyACM*), check
  the ESP32 is actually running the sketch (its onboard LED usually
  does *something* on boot), check baud rate matches (115200 both
  sides), check you're not trying to open the port while the Arduino
  IDE's Serial Monitor also has it open (only one program can at a time).
"""
import sys
import time

sys.path.insert(0, ".")
from serial_motors import SerialMotors


def stage1_link_test():
    print("=" * 60)
    print("STAGE 1: Serial link only (PING/PONG) -- no motors needed")
    print("=" * 60)
    motors = SerialMotors()
    if motors.available:
        print("[esp32] Serial link OK -- got PONG")
    else:
        print("[esp32] Serial link FAILED -- see error above. Fix this before Stage 2.")
    return motors


def run_for(seconds, label, motors):
    print(f"[motors] {label} for {seconds}s...")
    time.sleep(seconds)
    motors.stop()
    print("[motors] stopped")


def stage2_motor_menu(motors):
    print("=" * 60)
    print("STAGE 2: Motor exercise (robot must be on blocks!)")
    print("=" * 60)
    input("Confirm wheels are free to spin, press ENTER to continue... ")
    motors.set_speed(40)

    while True:
        print("\n1) Forward  2) Backward  3) Turn left  4) Turn right  5) Stop  q) Quit")
        choice = input("> ").strip().lower()
        if choice == '1':
            motors.forward(); run_for(2, "FORWARD", motors)
        elif choice == '2':
            motors.backward(); run_for(2, "BACKWARD", motors)
        elif choice == '3':
            motors.turn_left(); run_for(1.5, "TURN LEFT", motors)
        elif choice == '4':
            motors.turn_right(); run_for(1.5, "TURN RIGHT", motors)
        elif choice == '5':
            motors.stop()
        elif choice == 'q':
            motors.stop()
            break


if __name__ == "__main__":
    motors = stage1_link_test()
    if motors.available:
        proceed = input("\nSerial link works. Proceed to Stage 2 (motors)? [y/N] ").strip().lower()
        if proceed == 'y':
            stage2_motor_menu(motors)
