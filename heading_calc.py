import math
class Waypoint:
    def __init__(self, num, is_Active=0, relativity=3, type=16, params=[0,0,0,0], lat=0.0, long=0.0, alt=0.0, unknown=1):
        self.num = num
        self.is_Active = is_Active
        self.relativity = relativity
        self.type = type
        self.params = params
        self.lat = lat
        self.long = long
        self.alt = alt
        self.unknown = unknown
    
def calculate_heading(point1: Waypoint, point2: Waypoint) -> float:
    delta_long = point2.long - point1.long
    delta_lat = point2.lat - point1.lat
    heading = math.atan2(delta_long, delta_lat)
    return heading

def calculate_heading_xy(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)
