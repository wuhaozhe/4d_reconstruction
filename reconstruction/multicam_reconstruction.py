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

def reconstruct(args):
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

    for i in tqdm(range(len(rgb_video_list[0]))):
        tmp_color_image = [rgb_video_list[0][i], rgb_video_list[1][i], rgb_video_list[2][i]]
        tmp_depth_image = [depth_video_list[0][i], depth_video_list[1][i], depth_video_list[2][i]]
        lm = []
        for j in range(len(tmp_color_image)):
            face_lm = face_mesh.process(cv2.cvtColor(tmp_color_image[j], cv2.COLOR_BGR2RGB))
            if (face_lm.multi_face_landmarks is None) and j == 1:
                error_flag = True
            elif (face_lm.multi_face_landmarks is None) and j != 1:
                face_lm = None
            elif not (face_lm.multi_face_landmarks is None):
                face_lm = utils.mp_lm_2_np_lm(face_lm)
            lm.append(face_lm)

        if error_flag:
            break

        for j in range(args.camera_number):
            utils.crop_depth_image(tmp_depth_image[j], lm[j], cam_pinhole_list[j].width, cam_pinhole_list[j].height)    

        if i == 0:
            detect_lm_wo_occlude = lm[1][pointidx_wo_occlude[:, 0]]
            detect_lm_3d = utils.project_landmarks(detect_lm_wo_occlude, cam_pinhole_list[1], tmp_depth_image[1])
            
            detect_lm_w_occlude = lm[1][pointidx_w_occlude[:, 0]]
            detect_lm_2d = np.stack([detect_lm_w_occlude[:, 0] * cam_pinhole_list[1].width, detect_lm_w_occlude[:, 1] * cam_pinhole_list[1].height], axis = 1)

            pcd1, color1, normal1 = utils.create_pcd_from_rgbd(tmp_color_image[1], tmp_depth_image[1], cam_pinhole_list[1])
            mean_shape = np.matmul(mean_shape.cpu().numpy(), init_rot_mat)

            verts_x_scale = np.percentile(pcd1[:, 0], 95) - np.percentile(pcd1[:, 0], 5)
            bfm_x_scale = np.percentile(mean_shape[:, 0], 95) - np.percentile(mean_shape[:, 0], 5)
            scale = verts_x_scale / bfm_x_scale
            mean_shape = mean_shape * scale
            translation = np.mean(pcd1, axis = 0) - np.mean(mean_shape, axis = 0)
            deform_model.set_init_trans(scale, torch.from_numpy(init_rot_mat).to(args.device), torch.from_numpy(translation).to(args.device))

            # initial fitting
            optimizer_lm = bfm.construct_optimizer(['exp', 'identity', 'scale', 'rot_tensor', 'trans_tensor'], deform_model, lr = 1e-2)
            landmark_fitting(deform_model, optimizer_lm, torch.from_numpy(detect_lm_3d).to(args.device), torch.from_numpy(detect_lm_2d).to(args.device),
                            torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device), torch.from_numpy(pointidx_w_occlude[:, 1]).to(args.device), cam_pinhole_list[1], iter = 1000)

            gt_depth_image_list = []
            gt_color_image_list = []
            for j in range(3):
                gt_depth_image_list.append(torch.from_numpy(tmp_depth_image[j].astype(np.float32)).to(torch.device(args.device)) / 1000)
                gt_color_image_list.append(torch.from_numpy(tmp_color_image[j].astype(np.float32)).to(torch.device(args.device)).float() / 255)

            optimizer_coeff = bfm.construct_optimizer(['exp', 'identity', 'scale', 'rot_tensor', 'trans_tensor', 'gamma', 'tex'], deform_model, lr = 1e-2)
            loss_weight = {'lm_loss': 100, 'rgb_loss': 1, 'depth_loss': 2, 'reg_loss': 1e-3}
            depth_fitting_init(
                deform_model, 
                optimizer_coeff,
                torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                torch.from_numpy(detect_lm_3d).to(args.device), 
                cam_pinhole_list,
                diff_render,
                gt_depth_image_list,
                gt_color_image_list,
                [R0, R2, t0, t2],
                loss_weight,
                500    # iterations
            )
            
            optimizer_nicp = bfm.construct_optimizer(['nicp_trans'], deform_model, lr = 1e-2)
            loss_weight = {'lm_loss': 100, 'rgb_loss': 1, 'depth_loss': 2, 'reg_loss': 1e-3, 'laplacian_loss': 20, 'edge_loss': 20, 'offset_reg_loss': 0.01}
            depth_fitting_init(
                deform_model, 
                optimizer_nicp,
                torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                torch.from_numpy(detect_lm_3d).to(args.device), 
                cam_pinhole_list,
                diff_render,
                gt_depth_image_list,
                gt_color_image_list,
                [R0, R2, t0, t2],
                loss_weight,
                500    # iterations
            )

            for g in optimizer_nicp.param_groups:
                g['lr'] = 0.005
            
            for g in optimizer_coeff.param_groups:
                g['lr'] = 0.005

            depth_fitting_tune(
                deform_model, 
                optimizer_coeff,
                optimizer_nicp,
                torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                torch.from_numpy(detect_lm_3d).to(args.device), 
                cam_pinhole_list,
                diff_render,
                gt_depth_image_list,
                gt_color_image_list,
                [R0, R2, t0, t2],
                loss_weight,
                200    # iterations
            )

        else:
            detect_lm_wo_occlude = lm[1][pointidx_wo_occlude[:, 0]]
            detect_lm_3d = utils.project_landmarks(detect_lm_wo_occlude, cam_pinhole_list[1], tmp_depth_image[1])
            gt_depth_image_list = []
            gt_color_image_list = []
            for j in range(3):
                gt_depth_image_list.append(torch.from_numpy(tmp_depth_image[j].astype(np.float32)).to(torch.device(args.device)) / 1000)
                gt_color_image_list.append(torch.from_numpy(tmp_color_image[j].astype(np.float32)).to(torch.device(args.device)).float() / 255)

            loss_weight = {'lm_loss': 100, 'rgb_loss': 1, 'depth_loss': 2, 'reg_loss': 1e-3, 'laplacian_loss': 20, 'edge_loss': 20, 'offset_reg_loss': 0.01}
            depth_fitting_tune(
                deform_model, 
                optimizer_coeff,
                optimizer_nicp,
                torch.from_numpy(pointidx_wo_occlude[:, 1]).to(args.device),
                torch.from_numpy(detect_lm_3d).to(args.device), 
                cam_pinhole_list,
                diff_render,
                gt_depth_image_list,
                gt_color_image_list,
                [R0, R2, t0, t2],
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
    args.camera_number = 3
    args.bfm_folder = '../BFM'
    args.face_mask_path = '../test_data/bfm_mask.json'
    args.device = 'cuda'
    args.lm_mapping_path = '../test_data/landmark_mapping2.json'

    config.init_config(args)
    reconstruct(args)