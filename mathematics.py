from math import *


def clamp(value, minimum, maximum):
    return max(min(value, maximum), minimum)


class Vector2D:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __setattr__(self, key, value):
        self.__dict__[key] = float(value)  # All attributes will be floats.

    def __add__(self, other):
        x = self.x + other[0]
        y = self.y + other[1]
        return Vector2D(x, y)

    def __sub__(self, other):
        x = self.x - other[0]
        y = self.y - other[1]
        return Vector2D(x, y)

    def __mul__(self, other):
        if isinstance(other, Vector2D):
            x = self.x * other.x
            y = self.y * other.y
            return Vector2D(x, y)
        else:
            x = self.x * other
            y = self.y * other
            return Vector2D(x, y)

    def __mod__(self, other):
        x = self.x % other
        y = self.y % other
        return Vector2D(x, y)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, item):
        if   item == 0: return self.x
        elif item == 1: return self.y
        else: raise IndexError

    def __setitem__(self, key, value):
        if   key == 0: self.x = value
        elif key == 1: self.y = value
        else: raise IndexError

    def map(self, func, *args, **kwargs):
        x = func(self.x, *args, **kwargs)
        y = func(self.y, *args, **kwargs)
        return Vector2D(x, y)

    def as_int(self) -> tuple:
        x = int(self.x)
        y = int(self.y)
        return x, y

    def length(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def length_squared(self):
        return self.x ** 2 + self.y ** 2

    def dot(self, vector):
        return self.x * vector.x + self.y * vector.y

    def angle(self, vector, in_degrees=True):
        """
        a · b = ||a|| * ||b|| * cos 𝜽  <=>  cos 𝜽 = (a · b) / (||a|| * ||b||)  <=>  𝜽 = cos⁻¹ ((a · b) / (||a|| * ||b||))
        
        Args:
            vector: 
            in_degrees:

        Returns:
            Angle between two vectors.
        """
        scalar = self.dot(vector)
        lengths = self.length() * vector.length()
        angle = acos(scalar / lengths)

        if in_degrees:
            return degrees(angle)
        else:
            return angle

    def normalize(self):
        length = self.length()
        if length == 0: return Vector2D(0, 0)
        x = self.x / length
        y = self.y / length
        return Vector2D(x, y)

    def __repr__(self):
        return 'Vector2D({:.4}, {:.4})'.format(*self)


class Vector3D:

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __setattr__(self, key, value):
        self.__dict__[key] = float(value)  # All attributes will be floats.

    def __add__(self, other):
        x = self.x + other[0]
        y = self.y + other[1]
        z = self.z + other[2]
        return Vector3D(x, y, z)

    def __sub__(self, other):
        x = self.x - other[0]
        y = self.y - other[1]
        z = self.z - other[2]
        return Vector3D(x, y, z)

    def __mul__(self, other):
        if isinstance(other, Vector3D):
            x = self.x * other.x
            y = self.y * other.y
            z = self.z * other.z
            return Vector3D(x, y, z)
        elif isinstance(other, Matrix3D):
            raise TypeError('Operation order is invalid. The matrix must be on the left side of the expression.')
        else:
            x = self.x * other
            y = self.y * other
            z = self.z * other
            return Vector3D(x, y, z)

    def __mod__(self, other):
        x = self.x % other
        y = self.y % other
        z = self.z % other
        return Vector3D(x, y, z)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, item):
        if   item == 0: return self.x
        elif item == 1: return self.y
        elif item == 2: return self.z
        else: raise IndexError

    def __setitem__(self, key, value):
        if   key == 0: self.x = value
        elif key == 1: self.y = value
        elif key == 2: self.z = value
        else: raise IndexError

    def __len__(self):
        return 3

    def map(self, func, *args, **kwargs):
        x = func(self.x, *args, **kwargs)
        y = func(self.y, *args, **kwargs)
        z = func(self.z, *args, **kwargs)
        return Vector3D(x, y, z)

    def as_int(self) -> tuple:
        x = int(self.x)
        y = int(self.y)
        z = int(self.z)
        return x, y, z

    def length(self):
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        length = self.length()
        if length == 0: return Vector3D(0, 0, 0)
        x = self.x / length
        y = self.y / length
        z = self.z / length
        return Vector3D(x, y, z)

    def __repr__(self):
        return 'Vector3D({:.4}, {:.4}, {:.4})'.format(*self)



class Matrix3D:

    def __init__(self, row0=Vector3D(1, 0, 0), row1=Vector3D(0, 1, 0), row2=Vector3D(0, 0, 1)):
        self.row0 = Vector3D(*row0)
        self.row1 = Vector3D(*row1)
        self.row2 = Vector3D(*row2)

    def __add__(self, other):
        matrix = Matrix3D()
        for row in range(3):
            for col in range(3):
                matrix[row, col] = self[row, col] + other[row, col]
        return matrix

    def __sub__(self, other):
        matrix = Matrix3D()
        for row in range(3):
            for col in range(3):
                matrix[row, col] = self[row, col] - other[row, col]
        return matrix

    def __mul__(self, other):
        if isinstance(other, Matrix3D):
            matrix = Matrix3D()
            for row in range(3):
                for col in range(3):
                    matrix[row, col] = sum((
                        self[row, 0] * other[0, col],
                        self[row, 1] * other[1, col],
                        self[row, 2] * other[2, col]
                    ))
            return matrix
        elif isinstance(other, Vector3D):
            vector = Vector3D()
            for row in range(3):
                vector[row] = sum((
                    self[row, 0] * other[0],
                    self[row, 1] * other[1],
                    self[row, 2] * other[2]
                ))
            return vector
        else:
            matrix = Matrix3D()
            for row in range(3):
                for col in range(3):
                    matrix[row, col] = self[row, col] * other
            return matrix

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __iter__(self):
        yield self.row0
        yield self.row1
        yield self.row2

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
            if row == 0:
                return self.row0[col]
            elif row == 1:
                return self.row1[col]
            elif row == 2:
                return self.row2[col]
            else:
                raise IndexError
        elif item == 0:
            return self.row0
        elif item == 1:
            return self.row1
        elif item == 2:
            return self.row2
        else:
            raise IndexError

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            if row == 0:
                self.row0[col] = value
            elif row == 1:
                self.row1[col] = value
            elif row == 2:
                self.row2[col] = value
            else:
                raise IndexError
        elif key == 0:
            self.row0 = value
        elif key == 1:
            self.row1 = value
        elif key == 2:
            self.row2 = value
        else:
            raise IndexError

    def __repr__(self):
        return '| {}, {}, {} |\n| {}, {}, {} |\n| {}, {}, {} |\n'.format(*self.row0, *self.row1, *self.row2)

