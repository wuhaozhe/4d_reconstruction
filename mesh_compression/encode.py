import ffmpeg
import os
import math
from .utils import vertex2image

class SequenceEncoder(object):
    def __init__(self, save_path, vertex_number):
        self.stream_list = []
        self.save_path = save_path
        self.vertex_number = vertex_number
        image_size = math.ceil(math.sqrt(vertex_number + 4))
        for i in range(3):
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='gray16le', framerate=30, s='{}x{}'.format(image_size, image_size))
                .output(save_path + "_{}.nut".format(i), pix_fmt='gray16le', vcodec='ffv1', r=30, loglevel="quiet")
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            self.stream_list.append(process)

    def close(self):
        for i in range(3):
            self.stream_list[i].stdin.close()
            self.stream_list[i].wait()

    def delete_recording(self):
        for i in range(3):
            os.system('rm {}'.format(os.path.join(self.save_path, "_{}.nut".format(i))))

    def write_frame(self, vertex):
        '''
            vertex should be numpy array with shape (vertex_number * 3)
        '''
        compressed_image_list = vertex2image(vertex)
        for i in range(3):
            self.stream_list[i].stdin.write(compressed_image_list[i].tobytes())