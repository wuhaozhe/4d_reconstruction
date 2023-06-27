import utils
import torch

def point_boundary(in_points, k_neighbor = 100, alpha = 8):
    '''
        input: torch array with shape N * 3
        return bool array with size N
    '''
    neighbor_dis, neighbor_idx = utils.batch_nn(in_points, in_points, k = k_neighbor + 1)
    neighbor_dis = neighbor_dis[:, 1:]
    neighbor_idx = neighbor_idx[:, 1:]
    min_neighbor_dis = neighbor_dis[:, 0]
    neighbor_idx = neighbor_idx.reshape(-1)
    neighbor = in_points[neighbor_idx].reshape(-1, k_neighbor, 3)
    neighbor_mean = torch.mean(neighbor, dim = 1)
    diff = torch.norm(in_points - neighbor_mean, p = 2, dim = 1)
    edge_label = diff > alpha * torch.mean(min_neighbor_dis)
    return edge_label.detach()

def mesh_boundary(in_faces: torch.LongTensor, num_verts: int):
    '''
    input:
        in edges: N * 3, is the vertex index of each face, where N is number of faces
        num_verts: the number of vertexs mesh
    return:
        boundary_mask: bool tensor of num_verts, if true, point is on the boundary, else not
    '''
    in_x = in_faces[:, 0]
    in_y = in_faces[:, 1]
    in_z = in_faces[:, 2]
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
    output, count = torch.unique(in_hash, return_counts = True, dim = 0)
    boundary_edge = output[count == 1]
    boundary_vert1 = torch.div(boundary_edge, num_verts, rounding_mode='trunc')
    boundary_vert2 = boundary_edge % num_verts
    boundary_mask = torch.zeros(num_verts).bool().to(in_faces.device)
    boundary_mask[boundary_vert1] = True
    boundary_mask[boundary_vert2] = True
    return boundary_mask