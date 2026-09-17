# hw_bringup_tests

Phase 0 hardware bring-up, kept deliberately separate from `tracking_person/`.
Each script tests exactly one piece of hardware. Nothing here talks to
motors, ReID, or the other script's hardware. Goal: know for certain each
piece works on its own before debugging them combined.

## Before running either script

Copy this folder onto the Pi 5 (or `git clone`/`scp` it over), then adjust
two paths at the top of the scripts if yours differ from the defaults
(these match your `tracking_person` project's current config):

- `test_camera.py`: the `sys.path.append(...)` line pointing at your
  `pyorbbecsdk` build
- `test_coral.py`: `MODEL_PATH` / `MODEL_PATH_CPU` / `LABELS_PATH`

## Run order

### 1. `python3 test_camera.py`

Tests the Astra Pro only. Two windows should open (color + depth
colormap), and the console should print a live center distance.

**Pass:** distance reads roughly correct from ~40-60cm out and changes
sensibly as you move things closer/farther. Depth reading `0`/invalid
closer than that is expected (sensor blind zone), not a failure.

**If it fails:** fix this before touching anything else. Common causes —
USB3 port vs USB2/hub, `pyorbbecsdk` path wrong, camera not powered
enough through a hub.

### 2. `python3 test_coral.py`

Tests the Coral USB Accelerator only, using your plain webcam (not the
Astra) just to feed it *an* image. Point of this test is the TPU, not
the camera.

**Pass:** console prints `Coral Edge TPU delegate loaded successfully`,
then repeated inference times in milliseconds (should be low
single-digit-to-tens of ms on real TPU, much higher if it silently
fell back to CPU — the console tells you which).

**If it fails:** check the error message printed — it tells you whether
it's a missing Python package (wrong venv), missing model file, or the
delegate itself failing to load (usually USB power).

## After both pass

Only then move to combining camera + Coral detection (no motors yet),
per the phased plan — this is Phase 1, a separate step, not this folder.
