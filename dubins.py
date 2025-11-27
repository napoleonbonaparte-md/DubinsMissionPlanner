import ctypes
import os
import numpy as np

class PathPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("t", ctypes.c_float),
        ("valid", ctypes.c_bool),
        ("segment_idx", ctypes.c_int) # 0 - left, 1 - straight, 2 - right
    ]

class DubinsIterator:
    def __init__(self, start, goal, turning_radius, step_size=1.0):
        lib_path = os.path.join(os.path.dirname(__file__), "libdubins.so")
        self.lib = ctypes.CDLL(lib_path)
        # Constructor: DubinsIterator(float start[3], float goal[3], float turning_radius, float step_size)
        self.lib.create_dubins_iterator.restype = ctypes.c_void_p
        self.lib.create_dubins_iterator.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float]
        start_arr = (ctypes.c_float * 3)(*start)
        goal_arr = (ctypes.c_float * 3)(*goal)
        self.obj = self.lib.create_dubins_iterator(start_arr, goal_arr, ctypes.c_float(turning_radius), ctypes.c_float(step_size))
        # getNextPoint
        self.lib.dubins_iterator_get_next_point.restype = PathPoint
        self.lib.dubins_iterator_get_next_point.argtypes = [ctypes.c_void_p]
        # hasNext
        self.lib.dubins_iterator_has_next.restype = ctypes.c_bool
        self.lib.dubins_iterator_has_next.argtypes = [ctypes.c_void_p]
        # reset
        self.lib.dubins_iterator_reset.argtypes = [ctypes.c_void_p]
        # getAllPoints
        self.lib.dubins_iterator_get_all_points.restype = ctypes.POINTER(PathPoint)
        self.lib.dubins_iterator_get_all_points.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        # getSegmentPoints
        self.lib.dubins_iterator_get_segment_points.restype = ctypes.POINTER(PathPoint)
        self.lib.dubins_iterator_get_segment_points.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    def get_all_points(self):
        count = ctypes.c_int()
        ptr = self.lib.dubins_iterator_get_all_points(self.obj, ctypes.byref(count))
        if not ptr or count.value == 0:
            return []
        # Create a Python list from the C array
        return [ptr[i] for i in range(count.value)]
    def get_next_point(self):
        return self.lib.dubins_iterator_get_next_point(self.obj)

    def has_next(self):
        return self.lib.dubins_iterator_has_next(self.obj)

    def reset(self):
        self.lib.dubins_iterator_reset(self.obj)

    def get_segment_points(self):
        count = ctypes.c_int()
        ptr = self.lib.dubins_iterator_get_segment_points(self.obj, ctypes.byref(count))
        if not ptr or count.value == 0:
            return []
        return [ptr[i] for i in range(count.value)]

    def __del__(self):
        if hasattr(self, 'obj'):
            self.lib.destroy_dubins_iterator(self.obj)
