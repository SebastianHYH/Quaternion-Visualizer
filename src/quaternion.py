import numpy as np


class Quaternion:
    """A quaternion w + xi + yj + zk, used here to represent 3D rotations."""

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # Hamilton product
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        raise TypeError("Can only multiply a Quaternion by another Quaternion")

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return float(np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2))

    def normalize(self):
        """Return a unit quaternion. A zero quaternion falls back to identity."""
        n = self.norm()
        if n == 0:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def __repr__(self):
        return f"Quaternion({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def to_vector(self):
        """Return the vector part [x, y, z]."""
        return np.array([self.x, self.y, self.z])

    def to_wxyz(self):
        """Return all four components [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])

    # Kept for backwards compatibility; alias of to_vector.
    def to_array(self):
        return self.to_vector()

    @staticmethod
    def from_array(arr):
        """Build a quaternion from a 4-element [w, x, y, z] array."""
        if len(arr) != 4:
            raise ValueError("Array must have 4 elements [w, x, y, z]")
        return Quaternion(arr[0], arr[1], arr[2], arr[3])

    def to_axis_angle(self):
        """Return (axis unit vector, angle in degrees) for this rotation.

        For a near-zero rotation the axis is arbitrary; we return +Z.
        """
        q = self.normalize()
        angle = 2.0 * np.arccos(np.clip(q.w, -1.0, 1.0))
        sin_half = np.sqrt(max(0.0, 1.0 - q.w * q.w))
        if sin_half < 1e-8:
            return np.array([0.0, 0.0, 1.0]), 0.0
        axis = np.array([q.x, q.y, q.z]) / sin_half
        return axis, float(np.rad2deg(angle))

    def to_rotation_matrix(self):
        """Return the 3x3 rotation matrix for this quaternion (normalized first)."""
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])


def create_rotation_quaternion(axis, angle):
    """Build a unit rotation quaternion from an axis and an angle in degrees."""
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis cannot be zero")

    angle = np.deg2rad(angle)
    axis = np.asarray(axis, dtype=float) / norm
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    x = axis[0] * np.sin(half_angle)
    y = axis[1] * np.sin(half_angle)
    z = axis[2] * np.sin(half_angle)

    return Quaternion(w, x, y, z)


def rotate_point(point, axis, angle):
    """Rotate a single point about axis by angle (degrees) using q * p * q^-1."""
    q = create_rotation_quaternion(axis, angle)
    p = Quaternion(0, point[0], point[1], point[2])
    q_inv = q.conjugate()
    p_rotated = q * p * q_inv
    return p_rotated.to_vector()


def rotate_points(points, q):
    """Rotate an Nx3 array of points by quaternion q. Vectorized via the matrix."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points
    r = q.to_rotation_matrix()
    return points @ r.T


def slerp(q0, q1, t):
    """Shortest-path spherical linear interpolation between two unit quaternions.

    t in [0, 1]. t=0 returns q0, t=1 returns q1.
    """
    q0 = q0.normalize()
    q1 = q1.normalize()

    dot = float(np.dot(q0.to_wxyz(), q1.to_wxyz()))

    # Take the shorter arc: if the dot is negative, negate one quaternion.
    if dot < 0.0:
        q1 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
        dot = -dot

    # Very close together: fall back to normalized linear interpolation.
    if dot > 0.9995:
        result = Quaternion(
            q0.w + t * (q1.w - q0.w),
            q0.x + t * (q1.x - q0.x),
            q0.y + t * (q1.y - q0.y),
            q0.z + t * (q1.z - q0.z),
        )
        return result.normalize()

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0

    return Quaternion(
        s0 * q0.w + s1 * q1.w,
        s0 * q0.x + s1 * q1.x,
        s0 * q0.y + s1 * q1.y,
        s0 * q0.z + s1 * q1.z,
    )


def euler_to_matrix(roll, pitch, yaw, order="xyz"):
    """Build a rotation matrix from Euler angles (degrees).

    order gives the sequence of intrinsic axis rotations. The default "xyz"
    applies roll about X, then pitch about Y, then yaw about Z. This is a
    genuinely separate rotation representation from the quaternion path, so it
    exhibits gimbal lock at pitch = +/-90 degrees.
    """
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    mats = {"x": rx, "y": ry, "z": rz}
    result = np.eye(3)
    for axis in order:
        result = mats[axis] @ result
    return result


def from_euler(roll, pitch, yaw, order="xyz"):
    """Build a quaternion equivalent to the given Euler angles (degrees)."""
    axis_quats = {
        "x": lambda a: create_rotation_quaternion([1, 0, 0], a),
        "y": lambda a: create_rotation_quaternion([0, 1, 0], a),
        "z": lambda a: create_rotation_quaternion([0, 0, 1], a),
    }
    angles = {"x": roll, "y": pitch, "z": yaw}
    result = Quaternion(1.0, 0.0, 0.0, 0.0)
    for axis in order:
        result = axis_quats[axis](angles[axis]) * result
    return result
