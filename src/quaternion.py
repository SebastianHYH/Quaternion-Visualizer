import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    # Multiply
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        else:
            raise TypeError("Bukan quaternion")
    
    # Conjugate
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    # String
    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    # Array / Vector
    def to_array(self):
        return np.array([self.x, self.y, self.z])

    @staticmethod
    # Mengubah dari array ke Quaternion
    def from_array(arr):
        if len(arr) != 4:
            raise ValueError("Array perlu 4 elemen")
        return Quaternion(arr[0], arr[1], arr[2], arr[3])

# Membuat quaternion rotasi dari sumbu dan sudut
def create_rotation_quaternion(axis, angle):
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Axis tidak boleh nol")
    
    angle = np.deg2rad(angle) # Konversi ke radian
    axis = axis / norm
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    x = axis[0] * np.sin(half_angle)
    y = axis[1] * np.sin(half_angle)
    z = axis[2] * np.sin(half_angle)
    
    return Quaternion(w, x, y, z)

def rotate_point(point, axis, angle):
    q = create_rotation_quaternion(axis, angle)
    p = Quaternion(0, point[0], point[1], point[2])  # Point as quaternion
    q_inv = q.conjugate()
    p_rotated = q * p * q_inv  # Rotate point
    return p_rotated.to_array()
