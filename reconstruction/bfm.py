"""This script defines the parametric 3d face model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import os
import utils
import boundary
import json

class ParametricFaceModel:
    def __init__(self, 
                bfm_folder='./BFM', 
                recenter=True,
                default_name='BFM_model_front.mat'):
        
        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3*N,1]
        self._mean_shape = model['meanshape'].astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1

        self.meantex = model['meantex'].astype(np.float32)
        self.texBase = model['texBase'].astype(np.float32)
        
        if recenter:
            mean_shape = self._mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self._mean_shape = mean_shape.reshape([-1, 1])
            
        # self.vertex_mask = np.ones(self.point_buf.shape[0]).astype(bool)
        # self.face_mask = np.ones(self.face_buf.shape[0]).astype(bool)
        # if not (mask is None):
        #     self.vertex_mask[mask] = False
        #     v0_in_mask = np.isin(self.face_buf[:, 0], mask)
        #     v1_in_mask = np.isin(self.face_buf[:, 1], mask)
        #     v2_in_mask = np.isin(self.face_buf[:, 2], mask)
        #     self.face_mask[np.logical_or(np.logical_or(v0_in_mask, v1_in_mask), v2_in_mask)] = False
            # self.face_buf = self.face_buf[face_mask]

        self.device = 'cuda'

        self.a0 = np.pi
        self.a1 = 2 * np.pi / 1.7320508075688772
        self.a2 = 2 * np.pi / 2.8284271247461903
        self.c0 = 1 / np.sqrt(4 * np.pi)
        self.c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        self.c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        self.d0 = 0.5 / np.sqrt(3.0)
        self.Y0 = None


    
    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self._mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])


    def compute_norm(self, face_shape):
        with torch.no_grad():
            """
            Return:
                vertex_norm      -- torch.tensor, size (B, N, 3)

            Parameters:
                face_shape       -- torch.tensor, size (B, N, 3)
            """

            v1 = face_shape[:, self.face_buf[:, 0]]
            v2 = face_shape[:, self.face_buf[:, 1]]
            v3 = face_shape[:, self.face_buf[:, 2]]
            e1 = v1 - v2
            e2 = v2 - v3
            face_norm = torch.cross(e1, e2, dim=-1)
            face_norm = F.normalize(face_norm, dim=-1, p=2)
            face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
            
            vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
            vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
            return vertex_norm

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }


    def get_color(self, tex_coeff):
        n_b = tex_coeff.size(0)
        face_texture = torch.einsum(
            'ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        face_texture = face_texture.view(n_b, -1, 3)
        return face_texture


    def add_illumination(self, face_texture, norm, gamma):

        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8   # 避免颜色全黑

        gamma = gamma.permute(0, 2, 1)

        if self.Y0 is None or len(self.Y0) != n_v_full:
            self.Y0 = torch.ones(n_v_full).to(gamma.device).float() * self.a0 * self.c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(self.Y0)
        arrH.append(-self.a1 * self.c1 * ny)
        arrH.append(self.a1 * self.c1 * nz)
        arrH.append(-self.a1 * self.c1 * nx)
        arrH.append(self.a2 * self.c2 * nx * ny)
        arrH.append(-self.a2 * self.c2 * ny * nz)
        arrH.append(self.a2 * self.c2 * self.d0 * (3 * nz.pow(2) - 1))
        arrH.append(-self.a2 * self.c2 * nx * nz)
        arrH.append(self.a2 * self.c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

class DeformModel(nn.Module, ParametricFaceModel):
    def __init__(self, bfm_folder, face_mask_path, device = 'cuda', batch_size = 3):
        nn.Module.__init__(self)
        ParametricFaceModel.__init__(self, bfm_folder)
        self.device = torch.device(device)
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(self.device))

        self.batch_size = batch_size
        # deformation parameter
        self.exp = nn.Parameter(torch.zeros(1, 64).float().to(self.device))
        self.identity = nn.Parameter(torch.zeros(1, 80).float().to(self.device))
        self.gamma = nn.Parameter(torch.zeros(batch_size, 27).float().to(self.device))
        self.tex = nn.Parameter(torch.zeros(80).float().to(self.device))
        # self.nicp_A = nn.Parameter(torch.eye(3).unsqueeze(0).repeat(self.point_buf.shape[0], 1, 1).float().to(self.device))
        self.nicp_trans = nn.Parameter(torch.zeros(self.point_buf.shape[0], 3).float().to(self.device))
        self.edge = utils.edge_from_face(self.face_buf, self.point_buf.shape[0])

        # affine parameter
        self.scale = nn.Parameter(torch.ones(1).float().to(self.device))
        self.rot_tensor = nn.Parameter(torch.ones(4).float().to(self.device))
        self.trans_tensor = nn.Parameter(torch.zeros(3).float().to(self.device))

        # when find boundary points, we need to remove masked faces first
        face_mask_dict = json.load(open(face_mask_path, 'r'))
        face_mask = torch.tensor(face_mask_dict['left_eye'] + face_mask_dict['right_eye'] + face_mask_dict['nostril']).long().to(self.device)
        self.boundary_mask = boundary.mesh_boundary(self.face_buf, self.point_buf.shape[0])
        self.face_mask = self.boundary_mask.clone()
        self.face_mask[face_mask] = True
        

        # self.point_weight = torch.ones(self.point_buf.shape[0]).to(self.device) * 0.5


    def set_init_trans(self, scale, rot_matrix, translation):
        with torch.no_grad():
            self.scale.fill_(scale)
            self.rot_tensor[:] = utils.matrix_to_quaternion(rot_matrix)
            self.trans_tensor[:] = translation


    def regularization(self):
        reg_loss = torch.sum(self.exp**2) * 0.1 + torch.sum(self.identity**2) * 5 + torch.sum(self.gamma**2) * 0.1 + torch.sum(self.tex**2) * 0.1
        return reg_loss

    @property
    def mean_shape(self):
        mean_shape = self._mean_shape.clone().view(-1, 3)
        return mean_shape


    def forward(self, apply_offset = True):
        '''
            face_shape = Scale * (R * local_affine(BFM(exp, identity))) + translation
        '''
        face_shape = self.compute_shape(self.identity, self.exp).squeeze()
        # face_shape = face_shape.unsqueeze(2)
        # face_shape = torch.matmul(self.nicp_A, face_shape)
        if apply_offset:
            face_shape = face_shape + self.nicp_trans
        # face_shape.squeeze_()
        face_shape = torch.matmul(face_shape, utils.quaternion_to_matrix(self.rot_tensor))
        face_shape = self.scale * face_shape
        face_shape = face_shape + self.trans_tensor.unsqueeze(0)

        return face_shape

    def get_illuminated_color(self, face_shape):
        # face_shape is vertex with shape of batchsize * N * 3 (three images)
        face_norm = self.compute_norm(face_shape)
        face_tex = self.get_color(self.tex.unsqueeze(0)).repeat(self.batch_size, 1, 1)
        face_color = self.add_illumination(face_tex, face_norm, self.gamma)
        return face_color


def construct_optimizer(name_list, model, lr):
    params = list(filter(lambda kv: kv[0] in name_list, model.named_parameters()))
    params = list(map(lambda x: x[1], params))
    optimizer = torch.optim.Adam(params, lr = lr)
    
    return optimizer

def freeze_weight(model, name_list):
    params = list(filter(lambda kv: kv[0] in name_list, model.named_parameters()))
    params = list(map(lambda x: x[1], params))
    for param in params:
        param.requires_grad = False

def unfreeze_weight(model, name_list):
    params = list(filter(lambda kv: kv[0] in name_list, model.named_parameters()))
    params = list(map(lambda x: x[1], params))
    for param in params:
        param.requires_grad = True