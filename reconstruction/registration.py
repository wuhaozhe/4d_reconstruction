import torch
import utils
import cv2

def landmark_fitting(src_face_model, optimizer, target_vertex_3d, target_vertex_2d, idx_3d, idx_2d, pinhole_camera, iter = 100):
    
    for _ in range(iter):
        optimizer.zero_grad()
        
        # Forward pass
        deformed_vertex = src_face_model()
        src_vertex_3d = deformed_vertex[idx_3d]
        src_vertex_2d = deformed_vertex[idx_2d]
        src_vertex_2d_clip = utils.points_2_clip_space(src_vertex_2d, pinhole_camera)
        src_vertex_2d_ndcx = src_vertex_2d_clip[:, 0] / pinhole_camera.width
        src_vertex_2d_ndcy = src_vertex_2d_clip[:, 1] / pinhole_camera.height
        dst_vertex_2d_ndcx = target_vertex_2d[:, 0] / pinhole_camera.width
        dst_vertex_2d_ndcy = target_vertex_2d[:, 1] / pinhole_camera.height

        fit_loss = torch.sum((src_vertex_3d[target_vertex_3d[:, 2] > 0.1] - target_vertex_3d[target_vertex_3d[:, 2] > 0.1]).pow(2)) * 1000 + torch.sum((src_vertex_2d_ndcx - dst_vertex_2d_ndcx).pow(2)) * 500 +\
                    torch.sum((src_vertex_2d_ndcy - dst_vertex_2d_ndcy).pow(2)) * 500
        reg_loss = src_face_model.regularization() * 1e-3

        loss = reg_loss + fit_loss
        # Backward pass
        loss.backward()

        optimizer.step()

def _depth_fitting(src_face_model, lm_3d_idx, gt_lm_3d, cam_list, diff_render, gt_depth_img_list, gt_color_img_list, pose_list, loss_weight, iter_idx):
    deformed_vertex = src_face_model()
    deformed_vertex_trans_0 = torch.mm(deformed_vertex - pose_list[2], pose_list[0].T)
    deformed_vertex_trans_2 = torch.mm(deformed_vertex - pose_list[3], pose_list[1].T)
    vertex_all = torch.stack((deformed_vertex_trans_0, deformed_vertex, deformed_vertex_trans_2), dim = 0)
    face_color = src_face_model.get_illuminated_color(vertex_all)

    # weight = src_face_model.point_weight.unsqueeze(0).unsqueeze(2).repeat(3, 1, 1)
    face_mask = src_face_model.face_mask.unsqueeze(0).unsqueeze(2).repeat(3, 1, 1).float()
    deformed_vertex_ndc_0 = utils.points_2_ndc_space(deformed_vertex_trans_0, cam_list[0])
    deformed_vertex_ndc_1 = utils.points_2_ndc_space(deformed_vertex, cam_list[1])
    deformed_vertex_ndc_2 = utils.points_2_ndc_space(deformed_vertex_trans_2, cam_list[2])
    deformed_vertex_ndc_batch = torch.stack((deformed_vertex_ndc_0, deformed_vertex_ndc_1, deformed_vertex_ndc_2), dim = 0)

    depth = deformed_vertex_ndc_batch[:, :, 2].unsqueeze(2)
    # 后面记得加上texture color
    _, rast_image, pred_mask_image = diff_render(deformed_vertex_ndc_batch, src_face_model.face_buf.contiguous(),
        [cam_list[0].height, cam_list[0].width], torch.cat((depth, face_mask, face_color), dim = 2).contiguous()
    )
    pred_depth_image = rast_image[:, 0]     # shape B * H * W
    # weight_image = rast_image[:, 1]     # shape B * H * W
    face_mask_image = (1 - rast_image[:, 1]).bool()  # shape B * H * W
    color_image = rast_image[:, 2:]
    color_image = torch.flip(color_image.permute(0, 2, 3, 1), dims = [3]) / 255

    gt_depth_image = torch.stack(gt_depth_img_list, dim = 0)    # shape B * H * W
    gt_mask_image = (gt_depth_image > 0)
    gt_color_image = torch.stack(gt_color_img_list, dim = 0)    # shape B * H * W * 3


    # cv2.imwrite('test.png', weight_image[1].detach().cpu().numpy() * 128)
    mask_and = torch.logical_and(pred_mask_image.squeeze(), gt_mask_image)


    bfm_vertex = None
    
    # 比较大的问题是3dmm嘴唇不够厚，要通过nicp解决
    loss = 0.0
    loss_dict = {}

    # 要加一个相邻帧的正则项，minimize nicp的offset距离
    if 'lm_loss' in loss_weight:
        pred_lm_3d = src_face_model(apply_offset = False)[lm_3d_idx]
        lm_loss = torch.sum((pred_lm_3d[gt_lm_3d[:, 2] > 0.1] - gt_lm_3d[gt_lm_3d[:, 2] > 0.1]).pow(2)) * loss_weight['lm_loss']
        loss += lm_loss
        loss_dict['lm_loss'] = lm_loss.item()
    if 'rgb_loss' in loss_weight:
        color_diff = torch.sqrt(torch.sum(torch.pow((gt_color_image - color_image), 2), dim = 3))[mask_and]
        rgb_loss = torch.mean(utils.geman_mcclure(color_diff, 0.1)) * loss_weight['rgb_loss']
        loss += rgb_loss
        loss_dict['rgb_loss'] = rgb_loss.item()
    if 'depth_loss' in loss_weight:
        mask_tmp = torch.logical_and(face_mask_image, mask_and)
        diff = (pred_depth_image - gt_depth_image)[mask_tmp]
        depth_loss = torch.mean(utils.geman_mcclure(diff, 0.002)) * loss_weight['depth_loss']
        loss += depth_loss
        loss_dict['depth_loss'] = depth_loss.item()
    if 'reg_loss' in loss_weight:
        reg_loss = src_face_model.regularization() * loss_weight['reg_loss']
        loss += reg_loss
        loss_dict['reg_loss'] = reg_loss.item()
    if 'edge_loss' in loss_weight:
        if bfm_vertex is None:
            bfm_vertex = src_face_model(apply_offset = False)
            bfm_edge = src_face_model.edge
        bfm_edge_direction = bfm_vertex[bfm_edge[:, 0]] - bfm_vertex[bfm_edge[:, 1]]
        deform_edge_direction = deformed_vertex[bfm_edge[:, 0]] - deformed_vertex[bfm_edge[:, 1]]
        edge_loss = torch.sum((deform_edge_direction - bfm_edge_direction).pow(2)) * loss_weight['edge_loss']
        loss += edge_loss
        loss_dict['edge_loss'] = edge_loss.item()
        # print('edge', loss_dict['edge_loss'])
    if 'laplacian_loss' in loss_weight:
        # 参考权重为10(计算deformed vertex和3dmm的laplacian坐标距离)
        if bfm_vertex is None:
            bfm_vertex = src_face_model(apply_offset = False)
            bfm_edge = src_face_model.edge
        laplacian_matrix = utils.norm_laplacian(bfm_vertex, bfm_edge).detach()
        laplacian_matrix.requires_grad = False
        bfm_laplacian_ver = torch.sparse.mm(laplacian_matrix, bfm_vertex).detach()
        deform_laplacian_ver = torch.sparse.mm(laplacian_matrix, deformed_vertex)
        laplacian_loss = torch.sum((deform_laplacian_ver - bfm_laplacian_ver).pow(2)) * loss_weight['laplacian_loss']
        loss += laplacian_loss
        loss_dict['laplacian_loss'] = laplacian_loss.item()
        # print('lap', loss_dict['laplacian_loss'])
    if 'offset_reg_loss' in loss_weight:
        # 限制offset变动的幅度
        offset_reg_loss = torch.sum((src_face_model.nicp_trans).pow(2)) * loss_weight['offset_reg_loss']
        loss += offset_reg_loss
        loss_dict['offset_reg_loss'] = offset_reg_loss.item()

    loss.backward()

    if iter_idx % 50 == 0:
        cv2.imwrite('test1.png', color_image[1].detach().cpu().numpy() * 255)
        for key, value in loss_dict.items():
            print(key, value, end=' ')
        print()

def depth_fitting_tune(src_face_model, optimizer_coeff, optimizer_nicp, lm_3d_idx, gt_lm_3d, cam_list, diff_render, gt_depth_img_list, gt_color_img_list, pose_list, loss_weight, iter):
    for i in range(iter):
        optimizer_coeff.zero_grad()
        optimizer_nicp.zero_grad()
        _depth_fitting(src_face_model, lm_3d_idx, gt_lm_3d, cam_list, diff_render, gt_depth_img_list, gt_color_img_list, pose_list, loss_weight, i)
        optimizer_nicp.step()
        optimizer_coeff.step()

def depth_fitting_init(src_face_model, optimizer, lm_3d_idx, gt_lm_3d, cam_list, diff_render, gt_depth_img_list, gt_color_img_list, pose_list, loss_weight, iter):

    for i in range(iter):
        optimizer.zero_grad()
        _depth_fitting(src_face_model, lm_3d_idx, gt_lm_3d, cam_list, diff_render, gt_depth_img_list, gt_color_img_list, pose_list, loss_weight, i)
        optimizer.step()