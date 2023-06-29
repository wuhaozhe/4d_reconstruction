import numpy as np
import ffmpeg
import torch
import struct
import math

def decode_nut_to_np(video_path, vertex_num):
    vertex4d_coord = []
    image_size = math.ceil(math.sqrt(vertex_num + 4))
    for i in range(3):
        out, _ = (
            ffmpeg
            .input('{}_{}.nut'.format(video_path, i))
            .output('pipe:', format='rawvideo', pix_fmt='gray16le', loglevel="quiet")
            .run(capture_stdout=True)
        )
        video = np.frombuffer(out, np.uint16).reshape([-1, image_size * image_size])
        min_coord_array = np.zeros(len(video), dtype = np.float32)
        max_coord_array = np.zeros(len(video), dtype = np.float32)
        for j in range(len(video)):
            min_coord = struct.unpack('f', struct.pack('HH', video[j][0], video[j][1]))[0]
            max_coord = struct.unpack('f', struct.pack('HH', video[j][2], video[j][3]))[0]
            min_coord_array[j] = min_coord
            max_coord_array[j] = max_coord

        video_array = video[:, 4: 4 + vertex_num].astype(np.float32)
        if i == 0:
            vertex_array = np.zeros((len(video), vertex_num, 3), dtype = np.float32)
        vertex_array[:, :, i] = (video_array / 2**16) * (np.expand_dims(max_coord_array, axis = 1) - np.expand_dims(min_coord_array, axis = 1)) + np.expand_dims(min_coord_array, axis = 1)
        del out, _
    return vertex_array