import cv2
import utils
import camera
import open3d as o3d
import cv2
import numpy as np
import mediapipe as mp
import copy
import torch
mp_face_mesh = mp.solutions.face_mesh


def landmark_calibrate(lm, camera, depth_image):
    '''
        calibrate cameras according to facial landmarks
        return transformations which transform lm0 to lm1
    '''
    lm = copy.deepcopy(lm)
    lm0_3d = utils.project_landmarks(lm[0], camera[0], depth_image[0])
    lm1_3d = utils.project_landmarks(lm[1], camera[1], depth_image[1])

    for i in range(2):
        lm[i][:, 0] = lm[i][:, 0] * camera[i].width
        lm[i][:, 1] = lm[i][:, 1] * camera[i].height
        lm[i][:, 2] = lm[i][:, 2] * camera[i].width
    
    lm0_filtered_idx = utils.filter_error_projection(lm[0], lm0_3d)
    lm1_filtered_idx = utils.filter_error_projection(lm[1], lm1_3d)

    lmidx_intersect = np.intersect1d(lm0_filtered_idx, lm1_filtered_idx)

    lm0_3d_intersect = lm0_3d[lmidx_intersect]
    lm1_3d_intersect = lm1_3d[lmidx_intersect]
    

    # align 3d points
    R, t = utils.align_3d_points_np(lm0_3d_intersect, lm1_3d_intersect)

    # transed_src = np.matmul(lm0_3d_intersect, R) + t.reshape(1, -1)

    return R, t


def calibrate_three_camera(camera_intrinsic, icp_depth_list, icp_color_list, init_idx = 5, device = 'cuda'):
    '''
        calibrate the left and right camera to center camera

        input:
            calibrate_frame: three paths frames with calibrate plane of three cameras
            camera_intrinsic: three paths of yml files which contain the intrinsic of each camera
            icp_depth_list: list of human faces for icp registration, icp_depth_list can be list of file paths or numpy arrays
        return:
            camera extrinsics

        
        note that the camera distortion is not taken into consideration
    '''

    device = torch.device(device)

    # process input
    if isinstance(icp_depth_list[0], str):
        icp_depth_image = []
        for image_path in icp_depth_list:
            depth_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
            icp_depth_image.append(depth_image)
    else:
        icp_depth_image = icp_depth_list

    if isinstance(icp_color_list[0], str):
        icp_color_image = []
        for image_path in icp_color_list:
            color_image = cv2.imread(image_path)
            icp_color_image.append(color_image)
    else:
        icp_color_image = icp_color_list

    # process camera param
    cam_pinhole = []
    for i in range(3):
        cam_pinhole.append(camera.pinhole_camera(camera_intrinsic[i]))

    # coarse translation

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    lm = []
    for i in range(len(icp_color_image)):
        lm.append(utils.mp_lm_2_np_lm(face_mesh.process(cv2.cvtColor(icp_color_image[i], cv2.COLOR_BGR2RGB))))
    lm = np.array(lm)

    R0, t0 = landmark_calibrate([lm[0], lm[1]], [cam_pinhole[0], cam_pinhole[1]], [icp_depth_image[0], icp_depth_image[1]])
    R2, t2 = landmark_calibrate([lm[2], lm[1]], [cam_pinhole[2], cam_pinhole[1]], [icp_depth_image[2], icp_depth_image[1]])

    for i in range(len(icp_depth_image)):
        # 裁切背景点云，保留前景点云，将要裁切部分的深度值赋值为0
        utils.crop_depth_image(icp_depth_image[i], lm[i], cam_pinhole[0].width, cam_pinhole[0].height)
        utils.crop_depth_image(icp_depth_image[i], lm[i], cam_pinhole[1].width, cam_pinhole[1].height)
        utils.crop_depth_image(icp_depth_image[i], lm[i], cam_pinhole[2].width, cam_pinhole[2].height)

    radius = 0.002

    pcd0, color0, _ = utils.create_pcd_from_rgbd(icp_color_image[0], icp_depth_image[0], cam_pinhole[0])
    pcd0 = np.matmul(pcd0, R0) + np.expand_dims(t0, axis = 0)
    pcd1, color1, _ = utils.create_pcd_from_rgbd(icp_color_image[1], icp_depth_image[1], cam_pinhole[1])
    pcd2, color2, _ = utils.create_pcd_from_rgbd(icp_color_image[2], icp_depth_image[2], cam_pinhole[2])
    pcd2 = np.matmul(pcd2, R2) + np.expand_dims(t2, axis = 0)

    pcd0_o3d = utils.numpy_pcd_2_o3d_pcd(pcd0, color0)
    pcd1_o3d = utils.numpy_pcd_2_o3d_pcd(pcd1, color1)
    pcd2_o3d = utils.numpy_pcd_2_o3d_pcd(pcd2, color2)

    pcd0_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius * 2, max_nn = 30))
    pcd1_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius * 2, max_nn = 30))
    pcd2_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius * 2, max_nn = 30))

    result_icp_0 = o3d.pipelines.registration.registration_icp(
        pcd0_o3d, pcd1_o3d, radius, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=30))

    result_icp_2 = o3d.pipelines.registration.registration_icp(
        pcd2_o3d, pcd1_o3d, radius, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=30))


    R0_fine, t0_fine = result_icp_0.transformation[0:3, 0:3].T, result_icp_0.transformation[0:3, 3]
    R2_fine, t2_fine = result_icp_2.transformation[0:3, 0:3].T, result_icp_2.transformation[0:3, 3]

    R0_global = np.matmul(R0, R0_fine)
    t0_global = np.matmul(np.expand_dims(t0, axis = 0), R0_fine) + np.expand_dims(t0_fine, axis = 0)

    R2_global = np.matmul(R2, R2_fine)
    t2_global = np.matmul(np.expand_dims(t2, axis = 0), R2_fine) + np.expand_dims(t2_fine, axis = 0)

    # pcd0, color0, _ = utils.create_pcd_from_rgbd(icp_color_image[0], icp_depth_image[0], cam_pinhole[0])
    # pcd1, color1, _ = utils.create_pcd_from_rgbd(icp_color_image[1], icp_depth_image[1], cam_pinhole[1])
    # pcd2, color2, _ = utils.create_pcd_from_rgbd(icp_color_image[2], icp_depth_image[2], cam_pinhole[2])
    # pcd0 = np.matmul(pcd0, R0_global) + t0_global
    # pcd2 = np.matmul(pcd2, R2_global) + t2_global
    # pcd0_o3d = utils.numpy_pcd_2_o3d_pcd(pcd0, color0)
    # pcd1_o3d = utils.numpy_pcd_2_o3d_pcd(pcd1, color1)
    # pcd2_o3d = utils.numpy_pcd_2_o3d_pcd(pcd2, color2)

    # o3d.io.write_point_cloud('test_0.ply', pcd0_o3d)
    # o3d.io.write_point_cloud('test_1.ply', pcd1_o3d)
    # o3d.io.write_point_cloud('test_2.ply', pcd2_o3d)

    return [R0_global, t0_global, R2_global, t2_global]