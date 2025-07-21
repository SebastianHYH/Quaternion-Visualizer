import sys
import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from pyvistaqt import QtInteractor

from quaternion import rotate_point
from obj_loader import load_obj

class QuaternionVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quaternion Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Data
        self.vertices = np.array([])
        self.rotated_vertices = np.array([])
        self.mesh_original = None
        self.mesh_rotated = None
        self.axis_actor = None
        self.angle_label_actor = None

        # --- UI Setup ---
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # -- Kontrol (Panel Kiri) --
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(15)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # File Input
        self.btn_load_obj = QPushButton("Load .obj File")
        self.btn_load_obj.clicked.connect(self.load_file)
        self.lbl_file = QLabel("No file loaded.")
        controls_layout.addWidget(self.btn_load_obj)
        controls_layout.addWidget(self.lbl_file)

        # Axis Input
        controls_layout.addWidget(QLabel("Axis of Rotation (Unit Vector):"))
        self.txt_axis_x = QLineEdit("0")
        self.txt_axis_y = QLineEdit("1")
        self.txt_axis_z = QLineEdit("0")
        controls_layout.addWidget(self.txt_axis_x)
        controls_layout.addWidget(self.txt_axis_y)
        controls_layout.addWidget(self.txt_axis_z)

        # Angle Input
        controls_layout.addWidget(QLabel("Angle of Rotation (Degrees):"))
        self.txt_angle = QLineEdit("45")
        controls_layout.addWidget(self.txt_angle)
        
        # Tombol Rotasi
        self.btn_rotate = QPushButton("Apply Rotation")
        self.btn_rotate.clicked.connect(self.apply_rotation)
        controls_layout.addWidget(self.btn_rotate)
        
        # -- Visualisasi (Panel Kanan) --
        plotter_layout = QVBoxLayout()
        self.plotter = QtInteractor(self)
        plotter_layout.addWidget(self.plotter)

        # Gabungkan layout kontrol dan plotter
        main_layout.addLayout(controls_layout, 1) # 1/4 dari space
        main_layout.addLayout(plotter_layout, 3)  # 3/4 dari space
        
        self.setup_scene()

    def setup_scene(self):
        """Mengatur scene awal PyVista."""
        self.plotter.clear()
        self.plotter.add_axes() # Menampilkan sumbu koordinat XYZ
        self.plotter.add_text("Scene Initialized", position='upper_left', font_size=12)
        self.plotter.camera_position = 'iso'
        self.plotter.background_color = 'black'

    def load_file(self):
        """Membuka dialog untuk memilih file .obj."""
        filename, _ = QFileDialog.getOpenFileName(self, "Open .obj file", "", "OBJ Files (*.obj)")
        if filename:
            self.vertices = load_obj(filename)
            if self.vertices.size > 0:
                self.lbl_file.setText(f"Loaded: {filename.split('/')[-1]}")
                self.display_original_object()
            else:
                self.lbl_file.setText("Failed to load vertices.")
    
    def display_original_object(self):
        """Menampilkan objek asli sebelum rotasi."""
        self.setup_scene() # Reset scene
        
        # Hapus mesh lama jika ada
        if self.mesh_original:
            self.plotter.remove_actor(self.mesh_original)

        # Buat mesh dari vertices (wajah bisa diabaikan untuk visualisasi titik)
        cloud = pv.PolyData(self.vertices)
        self.mesh_original = self.plotter.add_mesh(
            cloud, 
            color='lightblue', 
            style='surface', 
            show_edges=True,
            label='Original'
        )
        self.plotter.add_legend()
        self.plotter.reset_camera()

    def apply_rotation(self):
        """Menerapkan kalkulasi dan visualisasi rotasi."""
        if self.vertices.size == 0:
            print("Please load an object first.")
            return

        try:
            # Ambil input dari user
            axis_x_text = self.txt_axis_x.text()
            axis_y_text = self.txt_axis_y.text()
            axis_z_text = self.txt_axis_z.text()
            angle_text = self.txt_angle.text()

            print(f"Axis: {axis_x_text}, {axis_y_text}, {axis_z_text}")
            print(f"Angle: {angle_text}")

            # Validasi input kosong
            if not axis_x_text or not axis_y_text or not axis_z_text or not angle_text:
                print("One or more input fields are empty. Please provide valid numbers.")
                return

            # Konversi input ke float
            axis = np.array([
                float(axis_x_text),
                float(axis_y_text),
                float(axis_z_text)
            ])
            angle_deg = float(angle_text)


            # Normalisasi axis untuk keamanan
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                print("Axis vector cannot be zero.")
                return
            axis_unit = axis / axis_norm

            # Lakukan rotasi untuk setiap vertex
            self.rotated_vertices = np.array([
                rotate_point(v, axis_unit, angle_deg) for v in self.vertices
            ])

            # Visualisasikan hasilnya
            self.display_rotated_object(axis_unit, angle_deg)

        except ValueError as e:
            print(f"Invalid input. Error: {e}. Please enter numbers for axis and angle.")

    def display_rotated_object(self, axis, angle):
        """Menampilkan objek setelah rotasi beserta axis dan angle."""
        # Tampilkan kembali objek asli
        self.display_original_object()

        # Hapus actor lama jika ada
        if self.mesh_rotated: self.plotter.remove_actor(self.mesh_rotated)
        if self.axis_actor: self.plotter.remove_actor(self.axis_actor)
        if self.angle_label_actor: self.plotter.remove_actor(self.angle_label_actor)

        # 1. Visualisasi Objek Setelah Rotasi
        rotated_cloud = pv.PolyData(self.rotated_vertices)
        self.mesh_rotated = self.plotter.add_mesh(
            rotated_cloud, 
            color='coral', 
            style='surface', 
            show_edges=True,
            label='Rotated'
        )

        # 2. Visualisasi Axis of Rotation
        # Buat garis panjang melewati titik origin
        axis_line_start = -2 * axis
        axis_line_end = 2 * axis
        self.axis_actor = self.plotter.add_lines(
            np.array([axis_line_start, axis_line_end]),
            color='green',
            width=5,
            label='Axis of Rotation'
        )
        
        # 3. Visualisasi Angle of Rotation
        # Tambahkan label di ujung axis
        label_pos = axis_line_end * 1.1
        self.angle_label_actor = self.plotter.add_point_labels(
            label_pos,
            [f"{angle}°"],
            font_size=16,
            point_color='green',
            text_color='green'
        )
        
        self.plotter.add_legend()
        self.plotter.reset_camera()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QuaternionVisualizer()
    window.show()
    sys.exit(app.exec())