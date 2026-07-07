import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from obj_loader import load_obj  # noqa: E402

TRIANGLE_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""

QUAD_OBJ = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""

INDEXED_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 3/1/1
"""


def _write(tmp_path, text):
    p = tmp_path / "model.obj"
    p.write_text(text)
    return str(p)


def test_triangle(tmp_path):
    verts, faces = load_obj(_write(tmp_path, TRIANGLE_OBJ))
    assert verts.shape == (3, 3)
    assert np.array_equal(faces, [3, 0, 1, 2])


def test_quad_is_fan_triangulated(tmp_path):
    verts, faces = load_obj(_write(tmp_path, QUAD_OBJ))
    assert verts.shape == (4, 3)
    # Quad -> two triangles: (0,1,2) and (0,2,3).
    assert np.array_equal(faces, [3, 0, 1, 2, 3, 0, 2, 3])


def test_indexed_face_forms(tmp_path):
    verts, faces = load_obj(_write(tmp_path, INDEXED_OBJ))
    assert verts.shape == (3, 3)
    # v/vt/vn tokens resolve to the vertex index only, 0-based.
    assert np.array_equal(faces, [3, 0, 1, 2])


def test_missing_file_returns_empty():
    verts, faces = load_obj("/nonexistent/path/model.obj")
    assert verts.shape == (0, 3)
    assert faces.size == 0


def test_no_vertices_returns_empty(tmp_path):
    verts, faces = load_obj(_write(tmp_path, "# just a comment\n"))
    assert verts.shape == (0, 3)
    assert faces.size == 0
