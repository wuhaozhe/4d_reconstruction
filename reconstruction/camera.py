import json


class pinhole_camera(object):
    def __init__(self, intrinsic_path):
        camera_intrinsic = json.load(open(intrinsic_path))
        self.width = camera_intrinsic['width']
        self.height = camera_intrinsic['height']
        self.fx = camera_intrinsic['intrinsic_matrix'][0]
        self.fy = camera_intrinsic['intrinsic_matrix'][4]
        self.cx = camera_intrinsic['intrinsic_matrix'][6]
        self.cy = camera_intrinsic['intrinsic_matrix'][7]