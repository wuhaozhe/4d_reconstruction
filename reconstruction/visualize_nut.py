import render
import os
import numpy as np
import pickle as pkl
import argparse
import sys
sys.path.append('..')
from tqdm import tqdm
from mesh_compression.decode import decode_nut_to_np

def visualize_sequence(vertex_array, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    faces = pkl.load(open('../test_data/bfm_faces.pkl', 'rb'))
    sequence_array[:, :, 2] *= -1
    position = np.mean(np.mean(sequence_array, axis = 0), axis = 0)
    position[2] = position[2] + 0.3

    for i in tqdm(range(len(sequence_array))):
        render.render_obj(vertex_array[i], faces, position, os.path.join(out_path, '{}.png'.format(i)))

    os.system("ffmpeg -framerate 30 -i {}/%d.png -b:v 200M -pix_fmt yuv420p {}/output.mp4".format(out_path, out_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required = True)
    parser.add_argument('--out_path', type=str, required = True)
    args = parser.parse_args()
    args.vertex_number = 35709

    sequence_array = decode_nut_to_np(args.file_path, args.vertex_number)
    visualize_sequence(sequence_array, args.out_path)