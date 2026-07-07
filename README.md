# Quaternion Visualizer

A desktop app for seeing what quaternion rotations actually do to a 3D object.
Load a `.obj` mesh, define a rotation three different ways, watch it animate, chain
rotations together, and compare quaternion interpolation against Euler angles to
see gimbal lock happen.

Built with PyQt6 (UI) + PyVista/VTK (3D rendering) + NumPy (math).

## Features

- **Real mesh rendering.** Loads vertices *and* faces from `.obj`, so objects show
  as solid shaded surfaces, not point clouds.
- **Three input modes** (tabs), all feeding one rotation:
  - Axis-Angle: rotation axis + angle in degrees.
  - Quaternion: raw `w, x, y, z` with a live normalized read-out.
  - Euler: roll / pitch / yaw.
- **Animated rotation.** Play button + scrub slider drive smooth `slerp`
  interpolation from the current orientation to the target.
- **Compose rotations.** "Apply" multiplies the new rotation onto the current
  orientation (quaternion multiplication). History list shows each step and the
  running composed quaternion; Undo steps back.
- **Gimbal-lock demo.** Toggle the compare mode to render the Euler-angle path
  (linear, yellow) next to the quaternion path (slerp, coral). They agree at the
  endpoints; push pitch toward 90 degrees and watch the Euler path diverge.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

Then click **Load sample cube** to get started (or **Load .obj File** for your own
mesh).

## How to use

1. Load an object.
2. Pick an input tab and enter a rotation. The normalized quaternion updates live.
3. **Apply** to compose it onto the current orientation. With **Animate** on it
   slerps into place; drag the slider to scrub the transition by hand.
4. Chain more rotations; use **Undo** / **Reset** to walk them back.
5. Tick **Compare Euler vs Quaternion**, enter Euler angles (try a large pitch),
   and **Play** to see gimbal lock.

## Tests

```bash
pytest
```

Deterministic unit tests cover the quaternion math (multiplication, conjugate,
rotation, slerp, matrix conversion, Euler equivalence) and the `.obj` loader
(triangles, fan-triangulated quads, `v/vt/vn` index forms, error paths). There is
no eval suite: the app does no LLM/latent-space work, so everything is covered by
these deterministic gate tests.

## Layout

```
run.py               # entry point (python run.py)
requirements.txt
assets/cube.obj      # sample mesh
src/
  main.py            # PyQt6 + PyVista GUI
  quaternion.py      # Quaternion class + rotation math (slerp, euler, matrix)
  obj_loader.py      # .obj vertex + face parser
tests/
  test_quaternion.py
  test_obj_loader.py
```
