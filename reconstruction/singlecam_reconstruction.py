import argparse
import os
import ffmpeg
import numpy as np
import calibrate
import config
import mediapipe as mp
import bfm
import torch
import json
import cv2
import utils
import sys
sys.path.append('..')
from camera import pinhole_camera
from tqdm import tqdm
from mesh_compression.encode import SequenceEncoder
from render import DiffMeshRender
from registration import landmark_fitting, depth_fitting_init, depth_fitting_tune
mp_face_mesh = mp.solutions.face_mesh