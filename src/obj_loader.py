import numpy as np


def _parse_face_index(token):
    """Parse one face token ('v', 'v/vt', 'v//vn', 'v/vt/vn') into a 0-based vertex index."""
    vertex_field = token.split("/")[0]
    return int(vertex_field) - 1  # .obj indices are 1-based


def load_obj(filename):
    """Load an .obj file.

    Returns (vertices, faces) where:
      - vertices is an Nx3 float array
      - faces is a flat int array in PyVista format: [n, i0, ..., n, j0, ...]
        n-gon faces are fan-triangulated. Empty if the file has no faces.

    On any error, returns (empty Nx3 array, empty array).
    """
    vertices = []
    faces = []
    try:
        with open(filename, "r") as file:
            for line in file:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append(list(map(float, parts[1:4])))
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    idx = [_parse_face_index(tok) for tok in parts]
                    # Fan-triangulate polygons with more than 3 vertices.
                    for i in range(1, len(idx) - 1):
                        faces.extend([3, idx[0], idx[i], idx[i + 1]])
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return np.empty((0, 3)), np.array([], dtype=np.int64)

    if not vertices:
        print("No vertices found in OBJ file.")
        return np.empty((0, 3)), np.array([], dtype=np.int64)

    return np.array(vertices, dtype=float), np.array(faces, dtype=np.int64)
