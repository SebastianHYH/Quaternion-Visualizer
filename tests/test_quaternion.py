import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quaternion import (  # noqa: E402
    Quaternion, create_rotation_quaternion, rotate_point, rotate_points, slerp,
    from_euler, euler_to_matrix,
)


def test_multiply_by_identity():
    q = Quaternion(0.5, 0.5, 0.5, 0.5)
    identity = Quaternion(1, 0, 0, 0)
    r = q * identity
    assert np.allclose(r.to_wxyz(), q.to_wxyz())


def test_conjugate():
    q = Quaternion(1, 2, 3, 4)
    c = q.conjugate()
    assert np.allclose(c.to_wxyz(), [1, -2, -3, -4])


def test_rotation_quaternion_is_unit():
    q = create_rotation_quaternion([0, 0, 1], 137.0)
    assert q.norm() == pytest.approx(1.0)


def test_rotate_point_90_about_z():
    out = rotate_point([1, 0, 0], [0, 0, 1], 90)
    assert np.allclose(out, [0, 1, 0], atol=1e-9)


def test_rotate_points_matches_loop():
    q = create_rotation_quaternion([1, 1, 0], 63.0)
    pts = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1]], dtype=float)
    vectorized = rotate_points(pts, q)
    axis, angle = q.to_axis_angle()
    looped = np.array([rotate_point(p, axis, angle) for p in pts])
    assert np.allclose(vectorized, looped, atol=1e-9)


def test_rotation_matrix_orthogonal():
    q = create_rotation_quaternion([0.3, -0.7, 0.5], 200.0)
    r = q.to_rotation_matrix()
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_slerp_endpoints():
    q0 = create_rotation_quaternion([0, 0, 1], 0)
    q1 = create_rotation_quaternion([0, 0, 1], 90)
    assert np.allclose(slerp(q0, q1, 0).to_wxyz(), q0.normalize().to_wxyz())
    assert np.allclose(slerp(q0, q1, 1).to_wxyz(), q1.normalize().to_wxyz())


def test_slerp_midpoint_is_unit_and_halfway():
    q0 = create_rotation_quaternion([0, 0, 1], 0)
    q1 = create_rotation_quaternion([0, 0, 1], 90)
    mid = slerp(q0, q1, 0.5)
    assert mid.norm() == pytest.approx(1.0)
    _, angle = mid.to_axis_angle()
    assert angle == pytest.approx(45.0, abs=1e-6)


def test_from_euler_matches_matrix():
    r, p, y = 30.0, 20.0, 45.0
    q = from_euler(r, p, y)
    assert np.allclose(q.to_rotation_matrix(), euler_to_matrix(r, p, y), atol=1e-9)


def test_wxyz_roundtrip():
    q = Quaternion(0.1, 0.2, 0.3, 0.4)
    assert np.allclose(Quaternion.from_array(q.to_wxyz()).to_wxyz(), q.to_wxyz())


def test_axis_angle_roundtrip():
    q = create_rotation_quaternion([1, 2, 3], 77.0)
    axis, angle = q.to_axis_angle()
    back = create_rotation_quaternion(axis, angle)
    assert np.allclose(back.to_wxyz(), q.to_wxyz(), atol=1e-9)


def test_normalize_zero_returns_identity():
    q = Quaternion(0, 0, 0, 0).normalize()
    assert np.allclose(q.to_wxyz(), [1, 0, 0, 0])
