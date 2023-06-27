import pickle as pkl
import numpy as np

faces = None

def init_config(args):
    global faces
    faces = np.array(pkl.load(open(args.faces_path, 'rb')))