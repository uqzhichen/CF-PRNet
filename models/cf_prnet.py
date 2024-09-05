import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

class TemplateMesh():

    def __init__(self, expand_d=0):
        self.expand_d = expand_d
        self.vertices, self.faces = self.create_3D_template_mesh()

    def get_vertices_faces(self):
        return self.vertices, self.faces

    def create_3D_template_mesh(self):
        ico = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.037)
        template = ico.subdivide_loop(number_of_iterations=self.expand_d)
        # o3d.io.write_triangle_mesh("coarse_object.obj", template)
        triangles = np.asarray(template.triangles)
        # Find triangles that reference the last two vertices
        mask = np.any(np.isin(triangles, [len(template.vertices) - 2, len(template.vertices) - 1]), axis=1)
        # Filter out these triangles
        filtered_triangles = triangles[~mask]
        # Set the filtered triangles back to the mesh
        template.triangles = o3d.utility.Vector3iVector(filtered_triangles)
        # Remove the last two vertices
        new_vertices = np.asarray(template.vertices)[:-2]  # Remove last two vertices
        # Set the new vertices back to the mesh
        template.vertices = o3d.utility.Vector3dVector(new_vertices)

        vertices = torch.from_numpy(np.asarray(
            template.vertices)).cuda().unsqueeze(0).float()
        faces = torch.from_numpy(np.asarray(
            template.triangles)).cuda().unsqueeze(0)

        return vertices, faces


class CFPRNet(nn.Module):
    """
    Coarse-toFine Prototype Refining Network for Shape Completion and Reconstruction

    Adapted from "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    """

    def __init__(self, params, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()
        self.params = params
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.params.coarse_p#  self.num_dense // (self.grid_size ** 2)

        self.zero_conv = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.05),
            nn.Conv1d(32, 64, 1)
        )

        self.first_conv = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.05),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.05),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.05),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.05),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

        template1 = TemplateMesh(self.params.coarse_d) # 5
        self.template1_points, self.template1_faces = template1.get_vertices_faces()
        self.template1_points = nn.Parameter(self.template1_points)
        # self.template1_points.requires_grad_(True)
        template2 = TemplateMesh(self.params.fine_d) # 7
        self.template2_points, self.template2_faces = template2.get_vertices_faces()
        self.template2_points = nn.Parameter(self.template2_points)
        # self.template2_points.requires_grad_(True)

        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(self.template2_points.detach().cpu().numpy()[0])
        # mesh.triangles = o3d.utility.Vector3iVector(self.template2_faces.detach().cpu().numpy()[0])
        # o3d.io.write_triangle_mesh("refined_fine_object.obj", mesh)

        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(self.template1_points.detach().cpu().numpy()[0])
        # mesh.triangles = o3d.utility.Vector3iVector(self.template1_faces.detach().cpu().numpy()[0])
        # o3d.io.write_triangle_mesh("refined_coarse_object.obj", mesh)
        self.offset_scaling = 1.0

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        feature = self.zero_conv(xyz.transpose(2, 1))                                        # (B,  64, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.first_conv(feature)                                                   # (B,  128, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)     # (B, num_coarse, 3), coarse point cloud
        coarse_output = torch.sigmoid(coarse) * self.offset_scaling * self.template1_points
        # coarse_output = (torch.tanh(coarse)+1) * self.offset_scaling * self.template1_points

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3)#.transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat.transpose(2, 1)], dim=1)                          # (B, 1024+2+3, num_fine)

        fine = self.final_conv(feat).transpose(2, 1) + point_feat
        fine_output = torch.sigmoid(fine) * self.offset_scaling * self.template2_points
        # fine_output = (torch.tanh(fine)+1) * self.offset_scaling * self.template2_points

        return coarse_output.contiguous(), fine_output.contiguous()
