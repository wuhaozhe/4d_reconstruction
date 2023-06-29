import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import trimesh
import pyrender
import nvdiffrast.torch as dr
import cv2
import torch
from torch import nn

def render_obj(vertex, faces, position, filename):
    vertex = vertex.copy()
    tmp = vertex[:, 0].copy()
    vertex[:, 0] = vertex[:, 1]
    vertex[:, 1] = tmp
    mesh = trimesh.Trimesh(vertices = vertex, faces = faces, vertex_colors = np.ones_like(vertex) * 128)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [0.0,  -1.0, 0.0, position[1]],
        [1.0,  0.0, 0.0, position[0]],
        [0.0,  0.0, 1.0, position[2]],
        [0.0,  0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=0.5,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(1000, 1000)
    color, _ = r.render(scene)
    cv2.imwrite(filename, color[:, :, ::-1])

# def render_objfile(objname, position, filename):
#     mesh = trimesh.load(objname, process=False)
#     mesh = pyrender.Mesh.from_trimesh(mesh)
#     scene = pyrender.Scene()
#     scene.add(mesh)
#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
#     camera_pose = np.array([
#         [0.0,  -1.0, 0.0, position[0]],
#         [1.0,  0.0, 0.0, position[1]],
#         [0.0,  0.0, 1.0, position[2]],
#         [0.0,  0.0, 0.0, 1.0],
#     ])
#     scene.add(camera, pose=camera_pose)
#     light = pyrender.SpotLight(color=np.ones(3), intensity=1,
#                                 innerConeAngle=np.pi/16.0,
#                                 outerConeAngle=np.pi/6.0)
#     scene.add(light, pose=camera_pose)
#     r = pyrender.OffscreenRenderer(1000, 1000)
#     color, _ = r.render(scene)
#     cv2.imwrite(filename, color[:, :, ::-1])


class DiffMeshRender(nn.Module):
    def __init__(self):
        super(DiffMeshRender, self).__init__()
        self.glctx = None

    def forward(self, vertex, tri, img_size, feat=None):
        '''
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, N, C), features
        '''

        device = vertex.device
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            # vertex[..., 1] = -vertex[..., 1]
        if self.glctx is None:
            self.glctx = dr.RasterizeCudaContext(device=device)
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex.contiguous(), tri, resolution = img_size)
        mask = (rast_out[..., 3] > 0).unsqueeze(1)
        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image
        
        return rast_out, image, mask