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
from registration import landmark_fitting, depth_fitting_init_singlecam, depth_fitting_tune_singlecam
mp_face_mesh = mp.solutions.face_mesh

def reconstruct(args):
    # load data
    intrinsics_path = args.file_path + "_{}.json".format(1)
    rgb_path = args.file_path + "_{}.mp4".format(1)
    depth_path = args.file_path + "_{}.nut".format(1)
    cam_pinhole = pinhole_camera(intrinsics_path)

    out, _ = (
        ffmpeg
        .input(rgb_path)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet")
        .run(capture_stdout=True)
    )
    rgb_video = np.frombuffer(out, np.uint8).reshape([-1, 1080, 1920, 3]).copy()

    out, _ = (
        ffmpeg
        .input(depth_path)
        .output('pipe:', format='rawvideo', pix_fmt='gray16le', loglevel="quiet")
        .run(capture_stdout=True)
    )
    depth_video = np.frombuffer(out, np.uint16).reshape([-1, 1080, 1920]).copy()

    if rgb_video.shape[0] != depth_video.shape[0]:
        raise Exception('rgb and depth length not consistent')

    init_rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # landmark detector
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # init parametric face model 
    deform_model = bfm.DeformModel(args.bfm_folder, args.face_mask_path, device = args.device, batch_size = 1)
    mean_shape = deform_model.mean_shape
    diff_render = DiffMeshRender()
    
    # init the mapping between mediapipe mesh and bfm mesh
    landmark_mapping = json.load(open(args.lm_mapping_path, 'r'))
    pointidx_wo_occlude = np.array(landmark_mapping['left_eyebrow'] + landmark_mapping['right_eyebrow'] + landmark_mapping['left_eye'] +\
                     landmark_mapping['right_eye'] + landmark_mapping['nose_bridge'] + landmark_mapping['outer_lip']).astype(np.int64)
    pointidx_w_occlude = np.array(landmark_mapping['nose_bottom'] + landmark_mapping['inner_lip'] + landmark_mapping['boundary']).astype(np.int64)

    
    # init mesh compression recorder
    compression_encoder = SequenceEncoder(args.save_path, 35709)

    # a flag to record whether error occurs in reconstruction
    error_flag = False

    lm = []
    for i in range(len(rgb_video)):
        color_image = rgb_video[i]
        face_lm = face_mesh.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        if face_lm.multi_face_landmarks is None:
            error_flag = True
        else:
            face_lm = utils.mp_lm_2_np_lm(face_lm)
        lm.append(face_lm)

        if error_flag:
            break

        utils.crop_depth_image(depth_video[i], face_lm, cam_pinhole.width, cam_pinhole.height)

    if error_flag:
        compression_encoder.delete_recording()
        compression_encoder.close()
        raise Exception('face not detected in video')
    else:
        for i in tqdm(range(len(rgb_video))):
            if i == 0:
                detect_lm_wo_occlude = lm[i][pointidx_wo_occlude[:, 0]]
                detect_lm_3d = utils.project_landmarks(detect_lm_wo_occlude, cam_pinhole, depth_video[i])

                detect_lm_w_occlude = lm[i][pointidx_w_occlude[:, 0]]
                detect_lm_2d = np.stack([detect_lm_w_occlude[:, 0] * cam_pinhole.width, detect_lm_w_occlude[:, 1] * cam_pinhole.height], axis = 1)

                pcd, color, normal = utils.create_pcd_from_rgbd(rgb_video[i], depth_video[i], cam_pinhole)
                mean_shape = np.matmul(mean_shape.cpu().numpy(), init_rot_mat)
                verts_x_scale = np.percentile(pcd[:, 0], 95) - np.percentile(pcd[:, 0], 5)
                bfm_x_scale = np.percentile(mean_shape[:, 0], 95) - np.percentile(mean_shape[:, 0], 5)
                scale = verts_x_scale / bfm_x_scale
                mean_shape = mean_shape * scale
                translation = np.mean(pcd, axis = 0) - np.mean(mean_shape, axis = 0)
                deform_model.set_init_trans(scale, torch.from_numpy(init_rot_mat).to(args.device), torch.from_numpy(translation).to(args.device))

                # initial fitting
                optimizer_lm = bfm.construct_optimizer(['exp', 'identity', 'scale', 'rot_tensor', 'trans_tensor'], deform_model, lr = 1e-2)
                landmark_fitting(deform_model, optimizer_lm, torch.from_numpy(detect_lm_3d).to(args.device), torch.from_numpy(detect_lm_2d).to(args.device),
                            torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device), torch.from_numpy(pointidx_w_occlude[:, 1]).to(args.device), cam_pinhole, iter = 1000)
                
                depth_tensor = torch.from_numpy(depth_video[i].astype(np.float32)).to(torch.device(args.device)) / 1000
                color_tensor = torch.from_numpy(rgb_video[i].astype(np.float32)).to(torch.device(args.device)) / 255
            
                optimizer_coeff = bfm.construct_optimizer(['exp', 'identity', 'scale', 'rot_tensor', 'trans_tensor', 'gamma', 'tex'], deform_model, lr = 1e-2)
                loss_weight = {'lm_loss': 100, 'rgb_loss': 1, 'depth_loss': 2, 'reg_loss': 1e-3}
                depth_fitting_init_singlecam(
                    deform_model, 
                    optimizer_coeff,
                    torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                    torch.from_numpy(detect_lm_3d).to(args.device), 
                    cam_pinhole,
                    diff_render,
                    depth_tensor,
                    color_tensor,
                    loss_weight,
                    500    # iterations
                )

                optimizer_nicp = bfm.construct_optimizer(['nicp_trans'], deform_model, lr = 1e-2)
                loss_weight = {'lm_loss': 100, 'rgb_loss': 1, 'depth_loss': 2, 'reg_loss': 1e-3, 'laplacian_loss': 20, 'edge_loss': 20, 'offset_reg_loss': 0.01}
                depth_fitting_init_singlecam(
                    deform_model, 
                    optimizer_nicp,
                    torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                    torch.from_numpy(detect_lm_3d).to(args.device), 
                    cam_pinhole,
                    diff_render,
                    depth_tensor,
                    color_tensor,
                    loss_weight,
                    500    # iterations
                )

                for g in optimizer_nicp.param_groups:
                    g['lr'] = 0.005
            
                for g in optimizer_coeff.param_groups:
                    g['lr'] = 0.005

                depth_fitting_tune_singlecam(
                    deform_model, 
                    optimizer_coeff,
                    optimizer_nicp,
                    torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                    torch.from_numpy(detect_lm_3d).to(args.device),
                    cam_pinhole,
                    diff_render,
                    depth_tensor,
                    color_tensor,
                    loss_weight,
                    200    # iterations
                )

            else:
                detect_lm_wo_occlude = lm[i][pointidx_wo_occlude[:, 0]]
                detect_lm_3d = utils.project_landmarks(detect_lm_wo_occlude, cam_pinhole, depth_video[i])
                depth_tensor = torch.from_numpy(depth_video[i].astype(np.float32)).to(torch.device(args.device)) / 1000
                color_tensor = torch.from_numpy(rgb_video[i].astype(np.float32)).to(torch.device(args.device)) / 255

                loss_weight = {'lm_loss': 100, 'rgb_loss': 1, 'depth_loss': 2, 'reg_loss': 1e-3, 'laplacian_loss': 20, 'edge_loss': 20, 'offset_reg_loss': 0.01}
                depth_fitting_tune_singlecam(
                    deform_model, 
                    optimizer_coeff,
                    optimizer_nicp,
                    torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                    torch.from_numpy(detect_lm_3d).to(args.device), 
                    cam_pinhole,
                    diff_render,
                    depth_tensor,
                    color_tensor,
                    loss_weight,
                    200    # iterations
                )

            src_face_shape = deform_model()
            compression_encoder.write_frame(src_face_shape.detach().cpu().numpy())
            # src_face_shape = utils.laplacian_smoothing(src_face_shape, deform_model.edge) * (1 - deform_model.boundary_mask.float().unsqueeze(1)) + \
            #                 src_face_shape * deform_model.boundary_mask.float().unsqueeze(1)
            # face_norm = deform_model.compute_norm(src_face_shape.unsqueeze(0))
            # face_tex = deform_model.get_color(deform_model.tex.unsqueeze(0))
            # face_color = deform_model.add_illumination(face_tex, face_norm, deform_model.gamma[1]) / 255
            # src_color = face_color.squeeze()

            # utils.write_obj_with_colors('../test_data/test.obj'.format(i), src_face_shape.detach().cpu().numpy(), deform_model.face_buf.cpu().numpy() + 1, src_color.detach().cpu().numpy()[:, ::-1])

    if error_flag:
        compression_encoder.delete_recording()
    # close compress encoder
    compression_encoder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required = True)
    parser.add_argument('--save_path', type=str, required = True)
    parser.add_argument('--faces_path', type=str, required = True)
    args = parser.parse_args()
    args.camera_number = 1
    args.bfm_folder = '../BFM'
    args.face_mask_path = '../test_data/bfm_mask.json'
    args.device = 'cuda'
    args.lm_mapping_path = '../test_data/landmark_mapping2.json'

    config.init_config(args)
    reconstruct(args)