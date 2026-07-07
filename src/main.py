import os
import sys

import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox, QTabWidget,
    QCheckBox, QSlider, QListWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor

from quaternion import (
    Quaternion, create_rotation_quaternion, rotate_points, slerp,
    from_euler, euler_to_matrix,
)
from obj_loader import load_obj

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


class QuaternionVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quaternion Visualizer")
        self.setGeometry(100, 100, 1300, 850)

        # --- Geometry ---
        self.base_points = np.empty((0, 3))
        self.faces = np.array([], dtype=np.int64)
        self.rot_poly = None
        self.euler_poly = None
        self.rot_actor = None
        self.orig_actor = None
        self.euler_actor = None
        self.axis_actor = None
        self.angle_label_actor = None

        # --- Rotation state ---
        self.current_orientation = Quaternion()   # composed orientation (identity)
        self.anim_from = Quaternion()
        self.anim_to = Quaternion()
        self.history = []                          # list of (label, Quaternion target)

        # --- Animation ---
        self.anim_t = 1.0
        self.timer = QTimer(self)
        self.timer.setInterval(16)                 # ~60 fps
        self.timer.timeout.connect(self._on_timer)

        self._build_ui()
        self.setup_scene()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        controls = QVBoxLayout()
        controls.setSpacing(10)
        controls.setAlignment(Qt.AlignmentFlag.AlignTop)

        controls.addWidget(self._build_object_group())
        controls.addWidget(self._build_input_group())
        controls.addWidget(self._build_apply_group())
        controls.addWidget(self._build_history_group())
        controls.addWidget(self._build_compare_group())

        plotter_layout = QVBoxLayout()
        self.plotter = QtInteractor(self)
        plotter_layout.addWidget(self.plotter)

        main_layout.addLayout(controls, 1)
        main_layout.addLayout(plotter_layout, 3)

    def _build_object_group(self):
        box = QGroupBox("Object")
        layout = QVBoxLayout(box)
        btn_load = QPushButton("Load .obj File")
        btn_load.clicked.connect(self.load_file)
        btn_sample = QPushButton("Load sample cube")
        btn_sample.clicked.connect(self.load_sample)
        self.lbl_file = QLabel("No file loaded.")
        layout.addWidget(btn_load)
        layout.addWidget(btn_sample)
        layout.addWidget(self.lbl_file)
        return box

    def _build_input_group(self):
        box = QGroupBox("Rotation input")
        layout = QVBoxLayout(box)
        self.tabs = QTabWidget()

        # Axis-Angle tab
        aa = QWidget()
        aa_layout = QGridLayout(aa)
        self.txt_axis_x = QLineEdit("0")
        self.txt_axis_y = QLineEdit("1")
        self.txt_axis_z = QLineEdit("0")
        self.txt_angle = QLineEdit("90")
        aa_layout.addWidget(QLabel("Axis X"), 0, 0)
        aa_layout.addWidget(self.txt_axis_x, 0, 1)
        aa_layout.addWidget(QLabel("Axis Y"), 1, 0)
        aa_layout.addWidget(self.txt_axis_y, 1, 1)
        aa_layout.addWidget(QLabel("Axis Z"), 2, 0)
        aa_layout.addWidget(self.txt_axis_z, 2, 1)
        aa_layout.addWidget(QLabel("Angle (deg)"), 3, 0)
        aa_layout.addWidget(self.txt_angle, 3, 1)
        self.tabs.addTab(aa, "Axis-Angle")

        # Quaternion tab
        q = QWidget()
        q_layout = QGridLayout(q)
        self.txt_qw = QLineEdit("1")
        self.txt_qx = QLineEdit("0")
        self.txt_qy = QLineEdit("0")
        self.txt_qz = QLineEdit("0")
        for i, (lbl, field) in enumerate([
            ("w", self.txt_qw), ("x", self.txt_qx), ("y", self.txt_qy), ("z", self.txt_qz)
        ]):
            q_layout.addWidget(QLabel(lbl), i, 0)
            q_layout.addWidget(field, i, 1)
        self.tabs.addTab(q, "Quaternion")

        # Euler tab
        e = QWidget()
        e_layout = QGridLayout(e)
        self.txt_roll = QLineEdit("0")
        self.txt_pitch = QLineEdit("0")
        self.txt_yaw = QLineEdit("0")
        for i, (lbl, field) in enumerate([
            ("Roll (X)", self.txt_roll), ("Pitch (Y)", self.txt_pitch), ("Yaw (Z)", self.txt_yaw)
        ]):
            e_layout.addWidget(QLabel(lbl), i, 0)
            e_layout.addWidget(field, i, 1)
        self.tabs.addTab(e, "Euler")

        layout.addWidget(self.tabs)
        self.lbl_readout = QLabel("q = (1.000, 0.000, 0.000, 0.000)")
        layout.addWidget(self.lbl_readout)

        # Live readout on any change.
        for field in [self.txt_axis_x, self.txt_axis_y, self.txt_axis_z, self.txt_angle,
                      self.txt_qw, self.txt_qx, self.txt_qy, self.txt_qz,
                      self.txt_roll, self.txt_pitch, self.txt_yaw]:
            field.textChanged.connect(self.update_readout)
        self.tabs.currentChanged.connect(self.update_readout)
        return box

    def _build_apply_group(self):
        box = QGroupBox("Apply / Animate")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply (compose)")
        self.btn_apply.clicked.connect(self.apply_rotation)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_orientation)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_reset)
        layout.addLayout(row)

        self.chk_animate = QCheckBox("Animate (slerp)")
        self.chk_animate.setChecked(True)
        layout.addWidget(self.chk_animate)

        anim_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_animation)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.on_slider)
        anim_row.addWidget(self.btn_play)
        anim_row.addWidget(self.slider)
        layout.addLayout(anim_row)
        return box

    def _build_history_group(self):
        box = QGroupBox("History (composed)")
        layout = QVBoxLayout(box)
        self.list_history = QListWidget()
        self.list_history.setMaximumHeight(110)
        layout.addWidget(self.list_history)
        row = QHBoxLayout()
        btn_undo = QPushButton("Undo")
        btn_undo.clicked.connect(self.undo_rotation)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.reset_orientation)
        row.addWidget(btn_undo)
        row.addWidget(btn_clear)
        layout.addLayout(row)
        return box

    def _build_compare_group(self):
        box = QGroupBox("Gimbal-lock demo")
        layout = QVBoxLayout(box)
        self.chk_compare = QCheckBox("Compare Euler (linear) vs Quaternion (slerp)")
        self.chk_compare.stateChanged.connect(self.on_compare_toggle)
        layout.addWidget(self.chk_compare)
        note = QLabel(
            "Uses the Euler tab angles, interpolated from zero.\n"
            "Yellow = linear Euler path, Coral = quaternion slerp.\n"
            "They agree at the ends; drive Pitch near 90 to see them\n"
            "diverge (gimbal lock)."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        return box

    # -------------------------------------------------------------- Scene ---
    def setup_scene(self):
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.camera_position = "iso"
        self.plotter.background_color = "black"
        self.rot_actor = None
        self.orig_actor = None
        self.euler_actor = None
        self.axis_actor = None
        self.angle_label_actor = None

    def _rebuild_actors(self):
        """Recreate mesh actors for the currently loaded geometry."""
        self.setup_scene()
        if self.base_points.size == 0:
            return

        orig_poly = pv.PolyData(self.base_points, self.faces)
        self.orig_actor = self.plotter.add_mesh(
            orig_poly, color="lightblue", style="wireframe", opacity=0.4, label="Original"
        )

        self.rot_poly = pv.PolyData(self.base_points.copy(), self.faces)
        self.rot_actor = self.plotter.add_mesh(
            self.rot_poly, color="coral", show_edges=True, label="Quaternion"
        )

        self.euler_poly = pv.PolyData(self.base_points.copy(), self.faces)
        self.euler_actor = self.plotter.add_mesh(
            self.euler_poly, color="yellow", show_edges=True, opacity=0.6, label="Euler"
        )
        self.euler_actor.SetVisibility(self.chk_compare.isChecked())

        self.plotter.reset_camera()

    # --------------------------------------------------------- Load files ---
    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open .obj file", "", "OBJ Files (*.obj)")
        if filename:
            self._load_path(filename, os.path.basename(filename))

    def load_sample(self):
        path = os.path.join(ASSETS_DIR, "cube.obj")
        if not os.path.exists(path):
            QMessageBox.warning(self, "Missing sample", f"Sample not found at {path}")
            return
        self._load_path(path, "cube.obj")

    def _load_path(self, path, name):
        vertices, faces = load_obj(path)
        if vertices.size == 0:
            QMessageBox.warning(self, "Load failed", f"No vertices found in {name}.")
            self.lbl_file.setText("Failed to load.")
            return
        if faces.size == 0:
            QMessageBox.information(
                self, "No faces",
                f"{name} has no face data; showing points only.",
            )
        self.base_points = vertices
        self.faces = faces
        self.lbl_file.setText(f"Loaded: {name} ({len(vertices)} verts)")
        self.reset_orientation()

    # ------------------------------------------------------- Input parsing ---
    def get_target_quaternion(self):
        """Read the active input tab and return a unit quaternion, or raise ValueError."""
        idx = self.tabs.currentIndex()
        if idx == 0:  # Axis-Angle
            axis = np.array([
                float(self.txt_axis_x.text()),
                float(self.txt_axis_y.text()),
                float(self.txt_axis_z.text()),
            ])
            if np.linalg.norm(axis) == 0:
                raise ValueError("Rotation axis cannot be zero.")
            return create_rotation_quaternion(axis, float(self.txt_angle.text()))
        if idx == 1:  # Quaternion
            q = Quaternion(
                float(self.txt_qw.text()), float(self.txt_qx.text()),
                float(self.txt_qy.text()), float(self.txt_qz.text()),
            )
            if q.norm() == 0:
                raise ValueError("Quaternion cannot be all zeros.")
            return q.normalize()
        # Euler
        return from_euler(
            float(self.txt_roll.text()), float(self.txt_pitch.text()), float(self.txt_yaw.text())
        )

    def get_euler_angles(self):
        return (
            float(self.txt_roll.text()),
            float(self.txt_pitch.text()),
            float(self.txt_yaw.text()),
        )

    def update_readout(self):
        try:
            q = self.get_target_quaternion()
            self.lbl_readout.setText(
                f"q = ({q.w:.3f}, {q.x:.3f}, {q.y:.3f}, {q.z:.3f})"
            )
        except (ValueError, TypeError):
            self.lbl_readout.setText("q = (invalid input)")

    # -------------------------------------------------------------- Apply ---
    def apply_rotation(self):
        if self.base_points.size == 0:
            QMessageBox.warning(self, "No object", "Load an object first.")
            return
        if self.chk_compare.isChecked():
            QMessageBox.information(
                self, "Demo mode",
                "Uncheck the gimbal-lock demo to compose rotations.",
            )
            return
        try:
            target = self.get_target_quaternion()
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return

        self.anim_from = self.current_orientation
        self.anim_to = (target * self.current_orientation).normalize()

        tab_name = self.tabs.tabText(self.tabs.currentIndex())
        self.history.append((tab_name, target))
        self._refresh_history()

        if self.chk_animate.isChecked():
            self._start_animation()
        else:
            self.current_orientation = self.anim_to
            self.slider.setValue(100)
            self._set_frame(1.0)

    def reset_orientation(self):
        self.timer.stop()
        self.current_orientation = Quaternion()
        self.anim_from = Quaternion()
        self.anim_to = Quaternion()
        self.history = []
        self._refresh_history()
        if self.base_points.size:
            self._rebuild_actors()
            self.slider.blockSignals(True)
            self.slider.setValue(100)
            self.slider.blockSignals(False)
            self._set_frame(1.0)

    def undo_rotation(self):
        if not self.history:
            return
        self.timer.stop()
        self.history.pop()
        # Recompose from identity.
        q = Quaternion()
        for _, target in self.history:
            q = (target * q).normalize()
        self.current_orientation = q
        self.anim_from = q
        self.anim_to = q
        self._refresh_history()
        self.slider.blockSignals(True)
        self.slider.setValue(100)
        self.slider.blockSignals(False)
        self._set_frame(1.0)

    def _refresh_history(self):
        self.list_history.clear()
        for name, target in self.history:
            self.list_history.addItem(
                f"{name}: ({target.w:.2f}, {target.x:.2f}, {target.y:.2f}, {target.z:.2f})"
            )
        q = self.current_orientation
        self.list_history.addItem(
            f"= composed ({q.w:.2f}, {q.x:.2f}, {q.y:.2f}, {q.z:.2f})"
        )

    # ---------------------------------------------------------- Animation ---
    def on_compare_toggle(self):
        on = self.chk_compare.isChecked()
        if self.euler_actor is not None:
            self.euler_actor.SetVisibility(on)
        self.btn_apply.setEnabled(not on)
        self.timer.stop()
        self.slider.blockSignals(True)
        self.slider.setValue(0 if on else 100)
        self.slider.blockSignals(False)
        self._set_frame(0.0 if on else 1.0)

    def play_animation(self):
        if self.base_points.size == 0:
            return
        if not self.chk_compare.isChecked():
            # Replay the last-applied transition.
            pass
        self._start_animation()

    def _start_animation(self):
        self.anim_t = 0.0
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.timer.start()

    def _on_timer(self):
        self.anim_t += 0.02
        if self.anim_t >= 1.0:
            self.anim_t = 1.0
            self.timer.stop()
            if not self.chk_compare.isChecked():
                self.current_orientation = self.anim_to
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.anim_t * 100))
        self.slider.blockSignals(False)
        self._set_frame(self.anim_t)

    def on_slider(self, value):
        self.timer.stop()
        self._set_frame(value / 100.0)

    def _set_frame(self, t):
        """Render the meshes at interpolation parameter t in [0, 1]."""
        if self.base_points.size == 0 or self.rot_poly is None:
            return

        if self.chk_compare.isChecked():
            try:
                roll, pitch, yaw = self.get_euler_angles()
            except (ValueError, TypeError):
                return
            q = slerp(Quaternion(), from_euler(roll, pitch, yaw), t)
            euler_m = euler_to_matrix(t * roll, t * pitch, t * yaw)
            self.euler_poly.points = self.base_points @ euler_m.T
        else:
            q = slerp(self.anim_from, self.anim_to, t)

        self.rot_poly.points = rotate_points(self.base_points, q)
        self._draw_axis(q)
        self.plotter.render()

    def _draw_axis(self, q):
        if self.axis_actor is not None:
            self.plotter.remove_actor(self.axis_actor)
        if self.angle_label_actor is not None:
            self.plotter.remove_actor(self.angle_label_actor)
        self.axis_actor = None
        self.angle_label_actor = None

        axis, angle = q.to_axis_angle()
        if angle < 1e-6:
            return
        extent = 1.5 * float(np.max(np.linalg.norm(self.base_points, axis=1)) or 1.0)
        self.axis_actor = self.plotter.add_lines(
            np.array([-extent * axis, extent * axis]), color="green", width=4
        )
        self.angle_label_actor = self.plotter.add_point_labels(
            (extent * 1.1 * axis).reshape(1, 3), [f"{angle:.1f}°"],
            font_size=14, point_color="green", text_color="green", show_points=False,
        )


def _force_x11_platform():
    """Route Qt through XWayland (xcb) on Linux unless the user overrode it.

    VTK's render window in this stack is vtkXOpenGLRenderWindow (X11-only). Under a
    Wayland session Qt defaults to the wayland platform plugin and hands VTK a
    Wayland surface, so VTK's X11 X_ConfigureWindow call fails with BadWindow and
    the app crashes on startup. Forcing xcb makes Qt and VTK agree on X11. This
    must run before QApplication is created (the plugin is chosen there).
    """
    if sys.platform.startswith("linux") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def main():
    _force_x11_platform()
    app = QApplication(sys.argv)
    window = QuaternionVisualizer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
