import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import config
from scipy.signal import butter, filtfilt
plt.switch_backend('agg')
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
cmap = plt.cm.viridis

def average_filter(face_pos_list, center_range, scale_range):
    face_pos_array = np.array(face_pos_list, dtype = np.float32)
    center_x = np.pad((face_pos_array[:, 0] + face_pos_array[:, 2]) / 2, center_range // 2, 'edge')
    center_y = np.pad((face_pos_array[:, 1] + face_pos_array[:, 3]) / 2, center_range // 2, 'edge')
    scale = np.pad(face_pos_array[:, 2] - face_pos_array[:, 0], scale_range // 2, 'edge')
    center_box = np.ones(center_range)/center_range
    scale_box = np.ones(scale_range)/scale_range
    center_x_smooth = np.convolve(center_x, center_box, mode='valid')
    center_y_smooth = np.convolve(center_y, center_box, mode='valid')
    scale_smooth = np.convolve(scale, scale_box, mode='valid') // 2
    x0 = center_x_smooth - scale_smooth
    y0 = center_y_smooth - scale_smooth
    x1 = center_x_smooth + scale_smooth
    y1 = center_y_smooth + scale_smooth
    rounded_pos = np.rint(np.stack((x0, y0, x1, y1, face_pos_array[:, -1]), axis = 1)).astype(int)
    return rounded_pos

def low_pass_filter(input_array, cutoff, fs, order=5):
    '''
        input array has shape of len * N
    '''
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    out_array = np.zeros_like(input_array)
    # import timeit
    # t1 = timeit.timeit()
    for i in range(input_array.shape[1]):
        out_array[:, i] = filtfilt(b, a, input_array[:, i])
    # t2 = timeit.timeit()
    # print(t2 - t1)
    # t = list(range(119))
    # plt.subplot(2, 1, 2)
    # plt.plot(t, input_array[:, 0], 'b-', label='data')
    # plt.plot(t, out_array[:, 0], 'g-', linewidth=2, label='filtered data')
    # plt.xlabel('Time [sec]')
    # plt.grid()
    # plt.legend()
    # plt.savefig('test.png')
    return out_array


def create_pcd_from_rgbd(rgb, depth, pinhole_camera, depth_scale = 1 / 1000):
    with torch.no_grad():

        # the unit of azure kinect is millimeter, we convert it to meter
        depth = torch.from_numpy(depth.astype(np.int32)).cuda()
        u = torch.arange(pinhole_camera.width).view(1, pinhole_camera.width).cuda()
        v = torch.arange(pinhole_camera.height).view(pinhole_camera.height, 1).cuda()
        z = depth * depth_scale
        x = (u - pinhole_camera.cx) * z / pinhole_camera.fx
        y = (v - pinhole_camera.cy) * z / pinhole_camera.fy
        points = torch.stack([x, y, z], dim = 2)
        x_prev = x[1:-1, :-2]
        x_next = x[1:-1, 2:]
        y_prev = y[:-2, 1:-1]
        y_next = y[2:, 1:-1]
        z_prev_x = z[1:-1, :-2]
        z_next_x = z[1:-1, 2:]
        z_prev_y = z[:-2, 1:-1]
        z_next_y = z[2:, 1:-1]
        dzdx = (z_next_x - z_prev_x) / (2.0 * (x_next - x_prev) + 1e-6)
        dzdy = (z_next_y - z_prev_y) / (2.0 * (y_next - y_prev) + 1e-6)
        dz_dx_mask = torch.logical_and(z_prev_x != 0, z_next_x != 0)
        dz_dy_mask = torch.logical_and(z_prev_y != 0, z_next_y != 0)
        normal_mask = torch.logical_and(dz_dx_mask, dz_dy_mask).unsqueeze(2)
        normal = torch.stack((dzdx, dzdy, torch.ones_like(dzdx).cuda()), dim = 2) * normal_mask
        normal = torch.nn.functional.pad(normal, (0, 0, 1, 1, 1, 1), mode = 'constant', value = 0)
        points = points.detach().cpu().numpy()
        normal = normal.detach().cpu().numpy()

    depth_mask = (depth != 0).cpu().numpy()
    masked_points = points[depth_mask]
    masked_colors = rgb[depth_mask]
    masked_normals = normal[depth_mask]
    magnitude = np.sqrt(masked_normals[:, 0]**2 + masked_normals[:, 1]**2 + masked_normals[:, 2]**2 + 1e-8)
    masked_normals = masked_normals / np.expand_dims(magnitude, axis = 1)
    return masked_points, masked_colors, masked_normals


def create_pcd_from_depth(depth, pinhole_camera, depth_scale = 1 / 1000):
    depth_mask = (depth != 0)
    u = np.arange(pinhole_camera.width).reshape(1, pinhole_camera.width)
    v = np.arange(pinhole_camera.height).reshape(pinhole_camera.height, 1)
    z = depth * depth_scale
    x = (u - pinhole_camera.cx) * z / pinhole_camera.fx
    y = (v - pinhole_camera.cy) * z / pinhole_camera.fy
    points = np.stack([x, y, z], axis = 2)
    masked_points = points[depth_mask]
    return masked_points


def project_landmarks(lm, pinhole_camera, depth_image, depth_scale = 1 / 1000):
    lm_x = (lm[:, 0] * pinhole_camera.width + 0.5).astype(np.int32)
    lm_y = (lm[:, 1] * pinhole_camera.height + 0.5).astype(np.int32)
    lm_index = lm_y * pinhole_camera.width + lm_x
    z = depth_image.reshape(-1)[lm_index] * depth_scale
    x = (lm_x - pinhole_camera.cx) * z / pinhole_camera.fx
    y = (lm_y - pinhole_camera.cy) * z / pinhole_camera.fy
    points = np.stack([x, y, z], axis = 1)

    return points

def points_2_clip_space(points, pinhole_camera, depth_scale = 1 / 1000):
    x = points[:, 0] * pinhole_camera.fx / points[:, 2] + pinhole_camera.cx
    y = points[:, 1] * pinhole_camera.fy / points[:, 2] + pinhole_camera.cy
    z = points[:, 2] / depth_scale

    return torch.stack([x, y, z], dim = 1)


def points_2_ndc_space(points, pinhole_camera, depth_scale = 1 / 1000):
    x = ((points[:, 0] * pinhole_camera.fx / points[:, 2] + pinhole_camera.cx) / pinhole_camera.width) * 2 - 1
    y = ((points[:, 1] * pinhole_camera.fy / points[:, 2] + pinhole_camera.cy) / pinhole_camera.height) * 2 - 1
    z = torch.clamp(points[:, 2], min = -1, max = 1)

    return torch.stack([x, y, z], dim = 1)



def filter_error_projection(lm_detect, lm_project):
    # filter the landmarks which cannot be projected correctly on pointclouds
    # return the filtered idx
    # 找到所有大致相似、并且法线方向也大致一致的三角形、并且法线与z轴夹角小于30度
    # 不用再额外过滤oval的点，因为相似三角形已经差不多过滤了
    faces = config.faces
    lm_detect_p0 = lm_detect[faces[:, 0]]
    lm_detect_p1 = lm_detect[faces[:, 1]]
    lm_detect_p2 = lm_detect[faces[:, 2]]
    
    lm_project_p0 = lm_project[faces[:, 0]]
    lm_project_p1 = lm_project[faces[:, 1]]
    lm_project_p2 = lm_project[faces[:, 2]]

    d_edge01 = lm_detect_p0 - lm_detect_p1
    d_edge02 = lm_detect_p0 - lm_detect_p2
    d_edge12 = lm_detect_p1 - lm_detect_p2

    norm_d_edge01 = d_edge01 / np.linalg.norm(d_edge01, axis = 1, keepdims = True)
    norm_d_edge02 = d_edge02 / np.linalg.norm(d_edge02, axis = 1, keepdims = True)
    norm_d_edge12 = d_edge12 / np.linalg.norm(d_edge12, axis = 1, keepdims = True)

    p_edge01 = lm_project_p0 - lm_project_p1
    p_edge02 = lm_project_p0 - lm_project_p2
    p_edge12 = lm_project_p1 - lm_project_p2

    norm_p_edge01 = p_edge01 / np.linalg.norm(p_edge01 + 1e-10, axis = 1, keepdims = True)
    norm_p_edge02 = p_edge02 / np.linalg.norm(p_edge02 + 1e-10, axis = 1, keepdims = True)
    norm_p_edge12 = p_edge12 / np.linalg.norm(p_edge12 + 1e-10, axis = 1, keepdims = True)
    

    d_angle_0 = np.arccos(np.sum(norm_d_edge01 * norm_d_edge02, axis = 1))
    d_angle_1 = np.arccos(np.sum(norm_d_edge01 * -1 * norm_d_edge12, axis = 1))
    d_angle_2 = np.arccos(np.sum(norm_d_edge02 * norm_d_edge12, axis = 1))

    p_angle_0 = np.arccos(np.sum(norm_p_edge01 * norm_p_edge02, axis = 1))
    p_angle_1 = np.arccos(np.sum(norm_p_edge01 * -1 * norm_p_edge12, axis = 1))
    p_angle_2 = np.arccos(np.sum(norm_p_edge02 * norm_p_edge12, axis = 1))

    normal_d = np.cross(norm_d_edge01, norm_d_edge02)
    normal_p = np.cross(norm_p_edge01, norm_p_edge02)

    angle_dis = np.abs(d_angle_0 - p_angle_0) + np.abs(d_angle_1 - p_angle_1) + np.abs(d_angle_2 - p_angle_2)
    normal_cos = np.sum(normal_d * normal_p, axis = 1)
    z_cos = normal_p[:, 2]

    filtered_triangles_idx = np.logical_and(np.logical_and((angle_dis < (np.pi / 6)), normal_cos >= 0.707), z_cos >= 0.707)

    vertex_idx = np.unique(faces[filtered_triangles_idx].reshape(-1))

    return vertex_idx

def homogeneous_translation(rotation, translation):
    translation_matrix = np.zeros((4, 4)).astype(np.float)
    translation_matrix[:3, :3] = rotation
    translation_matrix[:3, 3] = translation
    translation_matrix[3, 3] = 1.0
    return translation_matrix

def numpy_pcd_2_o3d_pcd(numpy_pcd, numpy_color = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(numpy_pcd)
    if not (numpy_color is None):
        pcd.colors = o3d.utility.Vector3dVector(numpy_color[:, ::-1] / 256)
    return pcd

def mp_lm_2_np_lm(lm):
    return np.array([[res.x, res.y, res.z] for res in lm.multi_face_landmarks[0].landmark]) if lm.multi_face_landmarks[0] else np.zeros(468, 3)

def align_3d_points_np(src, dst):
    src_center = src.mean(axis = 0)
    dst_center = dst.mean(axis = 0)
    src_norm = src - src_center
    dst_norm = dst - dst_center
    h_matrix = np.matmul(src_norm.T, dst_norm)

    U, S, Vh = np.linalg.svd(h_matrix)
    
    R = np.matmul(U, Vh)
    t = dst_center - np.matmul(R.T, np.expand_dims(src_center, axis = 1)).squeeze()

    return R, t


def align_3d_points_torch(src, dst):
    src_center = src.mean(dim = 0)
    dst_center = dst.mean(dim = 0)
    src_norm = src - src_center
    dst_norm = dst - dst_center
    h_matrix = torch.mm(src_norm.T, dst_norm)

    U, S, Vh = torch.linalg.svd(h_matrix)

    R = torch.mm(U, Vh)
    t = dst_center - torch.mm(R.T, src_center.unsqueeze(1)).squeeze()

    return R, t

def crop_depth_image(depth_image, lm, width, height, x_pad = 0, y_pad = 0, filter_threshold = 100):
    '''
        inplace crop of depth image
    '''
    if not (lm is None):
        lm_x = (lm[:, 0] * width + 0.5).astype(np.int32)
        lm_y = (lm[:, 1] * height + 0.5).astype(np.int32)
        min_x = max(np.min(lm_x) - x_pad, 0)
        max_x = min(np.max(lm_x) + x_pad, width)
        min_y = max(np.min(lm_y) - y_pad, 0)
        max_y = min(np.max(lm_y) + y_pad, height)
        mask = np.ones_like(depth_image, dtype = np.bool)
        mask[min_y: max_y, min_x: max_x] = False
        depth_image[mask] = 0
        min_value = np.min(depth_image[depth_image != 0])
        depth_image[depth_image > (min_value + filter_threshold)] = 0
    else:
        min_value = np.min(depth_image[depth_image != 0])
        depth_image[depth_image > (min_value + filter_threshold)] = 0

def batch_nn(in_vertex, query_vertex, batch_size = 1024, k = 1):
    topk_value = torch.zeros((len(query_vertex), k)).float().to(in_vertex.device)
    topk_idx = torch.zeros((len(query_vertex), k)).long().to(in_vertex.device)
    for i in range(0, len(query_vertex), batch_size):
        if i + batch_size >= len(query_vertex):
            batch_query_vertex = query_vertex[i:]
        else:
            batch_query_vertex = query_vertex[i: i + batch_size]

        diff = torch.norm(batch_query_vertex.unsqueeze(1) - in_vertex.unsqueeze(0), p = 2, dim = 2)
        tmp_topk_value, tmp_topk_idx = torch.topk(diff, k, dim = 1, largest = False)
        
        if i + batch_size >= len(query_vertex):
            topk_value[i:] = tmp_topk_value
            topk_idx[i:] = tmp_topk_idx
        else:
            topk_value[i: i + batch_size] = tmp_topk_value
            topk_idx[i: i + batch_size] = tmp_topk_idx
    return topk_value, topk_idx

def edge_from_face(input_face, num_verts):
    '''
        input_face has shape of N * 3
        output edge with M * 2
    '''
    in_x = input_face[:, 0]
    in_y = input_face[:, 1]
    in_z = input_face[:, 2]
    in_xy = in_x * (num_verts) + in_y
    in_yx = in_y * (num_verts) + in_x
    in_xz = in_x * (num_verts) + in_z
    in_zx = in_z * (num_verts) + in_x
    in_yz = in_y * (num_verts) + in_z
    in_zy = in_z * (num_verts) + in_y
    in_xy_hash = torch.minimum(in_xy, in_yx)
    in_xz_hash = torch.minimum(in_xz, in_zx)
    in_yz_hash = torch.minimum(in_yz, in_zy)
    in_hash = torch.cat((in_xy_hash, in_xz_hash, in_yz_hash), dim = 0)
    output, _ = torch.unique(in_hash, return_counts = True, dim = 0)
    vert1 = torch.div(output, num_verts, rounding_mode='trunc')
    vert2 = output % num_verts
    edge = torch.cat((vert1.unsqueeze(1), vert2.unsqueeze(1)), dim = 1)
    return edge

def matrix_to_quaternion(matrix):
    '''
        input torch tensor with shape 3 * 3
    '''
    tr = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (matrix[2, 1] - matrix[1, 2]) / S
        qy = (matrix[0, 2] - matrix[2, 0]) / S
        qz = (matrix[1, 0] - matrix[0, 1]) / S
    elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
        S = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        qw = (matrix[2, 1] - matrix[1, 2]) / S
        qx = 0.25 * S
        qy = (matrix[0, 1] + matrix[1, 0]) / S
        qz = (matrix[0, 2] + matrix[2, 0]) / S
    elif (matrix[1, 1] > matrix[2, 2]):
        S = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        qw = (matrix[0, 2] - matrix[2, 0]) / S
        qx = (matrix[0, 1] + matrix[1, 0]) / S
        qy = 0.25 * S
        qz = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        qw = (matrix[1, 0] - matrix[0, 1]) / S
        qx = (matrix[0, 2] + matrix[2, 0]) / S
        qy = (matrix[1, 2] + matrix[2, 1]) / S
        qz = 0.25 * S

    return torch.tensor([qw, qx, qy, qz]).to(matrix.device)


def quaternion_to_matrix(q):
    '''
        input torch tensor with shape 4
        四元数必须是标准化后的, 在计算matrix前先做一次标准化
    '''
    q = torch.nn.functional.normalize(q, p = 2, dim = 0)
    s, x, y, z = q[0], q[1], q[2], q[3]
    m00 = 1 - 2 * y * y - 2 * z * z
    m01 = 2 * x * y - 2 * s * z
    m02 = 2 * x * z + 2 * s * y
    m10 = 2 * x * y + 2 * s * z
    m11 = 1 - 2 * x * x - 2 * z * z
    m12 = 2 * y * z - 2 * s * x
    m20 = 2 * x * z - 2 * s * y
    m21 = 2 * y * z + 2 * s * x
    m22 = 1 - 2 * x * x - 2 * y * y
    return torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22]).view(3, 3)

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    triangles = triangles.copy() # meshlab start with 1
    triangles = triangles.T
    vertices = vertices.T

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f} {} {} {}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i], colors[i, 2],
                                               colors[i, 1], colors[i, 0])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[0, i], triangles[1, i], triangles[2, i])
            f.write(s)

def colored_depthmap(depth, d_min=0, d_max=1):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return (255 * cmap(depth_relative)[:,:,:3]).astype('uint8') # HWC


def norm_laplacian(
    verts: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Norm laplacian computes a variant of the laplacian matrix which weights each
    affinity with the normalized distance of the neighboring nodes.
    More concretely,
    L[i, j] = 1. / wij where wij = ||vi - vj|| if (vi, vj) are neighboring nodes

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    edge_verts = verts[edges]  # (E, 2, 3)
    v0, v1 = edge_verts[:, 0], edge_verts[:, 1]

    # Side lengths of each edge, of shape (E,)
    w01 = 1.0 / ((v0 - v1).norm(dim=1) + eps)
    w01 = w01 / torch.mean(w01)
    w01 = torch.sqrt(w01)

    # Construct a sparse matrix by basically doing:
    # L[v0, v1] = w01
    # L[v1, v0] = w01
    e01 = edges.t()  # (2, E)

    V = verts.shape[0]
    L = torch.sparse.FloatTensor(e01, w01, (V, V))
    L = L + L.t()

    diagonal = -1 * torch.sparse.sum(L, dim = 1).to_dense()
    pos = torch.arange(V).to(diagonal.device).unsqueeze(0).repeat(2, 1)
    diagonal = torch.sparse.FloatTensor(pos, diagonal, (V, V))
    L = L + diagonal

    return L

def laplacian_smoothing(verts: torch.Tensor, edges: torch.Tensor):
    V = verts.shape[0]
    e01 = edges.t()
    L = torch.sparse.FloatTensor(e01, torch.ones(len(edges)).to(e01), (V, V)).float()
    L = L + L.t()
    neighbor_num = torch.sparse.sum(L, dim = 1).to_dense()
    verts_smooth = torch.sparse.mm(L, verts).to_dense() / neighbor_num.unsqueeze(1)
    return verts_smooth


def geman_mcclure(x, scale):
    scaled_sqaure_x = (x / scale)**2
    loss = scaled_sqaure_x * 2 / (scaled_sqaure_x + 4)

    return loss