import numpy as np
import math
import struct

def vertex2image(vertex, pad = 0.05):
    # vertex has shape of N * 3, where N is the number of vertex
    image_list = []
    image_size = math.ceil(math.sqrt(vertex.shape[0] + 4))
    for i in range(3):
        vertex_axis = vertex[:, i]
        min_coord = np.min(vertex_axis) - pad
        max_coord = np.max(vertex_axis) + pad
        vertex_normalized = (vertex_axis - min_coord) / (max_coord - min_coord)
        vertex_quant = np.around(vertex_normalized * 2**16).astype(np.uint16)
        min_coord_uint16 = struct.unpack('HH', struct.pack('f', min_coord))
        max_coord_uint16 = struct.unpack('HH', struct.pack('f', max_coord))
        saved_image = np.zeros(image_size**2, dtype = np.uint16).reshape(-1)
        saved_image[0: 2] = min_coord_uint16
        saved_image[2: 4] = max_coord_uint16
        saved_image[4: 4 + len(vertex_quant)] = vertex_quant
        saved_image = saved_image.reshape(image_size, image_size)
        image_list.append(saved_image)
    
    return image_list