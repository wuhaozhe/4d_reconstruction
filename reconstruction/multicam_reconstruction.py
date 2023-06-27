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
import sys
sys.path.append('..')
from camera import pinhole_camera
from mesh_compression.encode import SequenceEncoder
mp_face_mesh = mp.solutions.face_mesh

def reconstruct(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    file_suffix = args.file_path.split('/')[-1]


    # load data
    intrinsics_path = []
    for i in range(args.camera_number):
        intrinsics_path.append(args.file_path + "_{}.json".format(i))

    rgb_path = []
    for i in range(args.camera_number):
        rgb_path.append(args.file_path + "_{}.mp4".format(i))

    depth_path = []
    for i in range(args.camera_number):
        depth_path.append(args.file_path + "_{}.nut".format(i))

    cam_pinhole_list = []
    for i in range(args.camera_number):
        cam_pinhole_list.append(pinhole_camera(intrinsics_path[i]))

    rgb_video_list = []
    for i in range(args.camera_number):
        out, _ = (
            ffmpeg
            .input(rgb_path[i])
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet")
            .run(capture_stdout=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, 1080, 1920, 3]).copy()
        rgb_video_list.append(video)

    depth_video_list = []
    for i in range(args.camera_number):
        out, _ = (
            ffmpeg
            .input(depth_path[i])
            .output('pipe:', format='rawvideo', pix_fmt='gray16le', loglevel="quiet")
            .run(capture_stdout=True)
        )
        video = np.frombuffer(out, np.uint16).reshape([-1, 1080, 1920]).copy()
        depth_video_list.append(video)

    # calibrate cameras
    icp_depth_list = [depth_video_list[0][0], depth_video_list[1][0], depth_video_list[2][0]]
    icp_color_list = [rgb_video_list[0][0], rgb_video_list[1][0], rgb_video_list[2][0]]
    R0, t0, R2, t2 = calibrate.calibrate_three_camera(intrinsics_path, icp_depth_list, icp_color_list, device = args.device)
    R0 = torch.from_numpy(R0).float().to(torch.device(args.device))
    t0 = torch.from_numpy(t0).float().to(torch.device(args.device))
    R2 = torch.from_numpy(R2).float().to(torch.device(args.device))
    t2 = torch.from_numpy(t2).float().to(torch.device(args.device))
    init_rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # landmark detector
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # init parametric face model 
    deform_model = bfm.DeformModel(args.bfm_folder, args.face_mask_path, device = args.device)
    mean_shape = deform_model.mean_shape
    
    # init the mapping between mediapipe mesh and bfm mesh
    landmark_mapping = json.load(open(args.lm_mapping_path, 'r'))
    pointidx_wo_occlude = np.array(landmark_mapping['left_eyebrow'] + landmark_mapping['right_eyebrow'] + landmark_mapping['left_eye'] +\
                     landmark_mapping['right_eye'] + landmark_mapping['nose_bridge'] + landmark_mapping['outer_lip']).astype(np.int64)
    pointidx_w_occlude = np.array(landmark_mapping['nose_bottom'] + landmark_mapping['inner_lip'] + landmark_mapping['boundary']).astype(np.int64)

    
    # init mesh compression recorder
    compression_encoder = SequenceEncoder(args.save_path, 35709)


    # a flag to record whether error occurs in reconstruction
    error_flag = False


    # close compress encoder
    compression_encoder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required = True)
    parser.add_argument('--save_path', type=str, required = True)
    parser.add_argument('--faces_path', type=str, required = True)
    args = parser.parse_args()
    args.camera_number = 3
    args.bfm_folder = '../BFM'
    args.face_mask_path = '../test_data/bfm_mask.json'
    args.device = 'cuda'
    args.lm_mapping_path = '../test_data/landmark_mapping2.json'

    config.init_config(args)
    reconstruct(args)