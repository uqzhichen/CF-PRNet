import os
import argparse

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data
from dataset.fruit import IGGFruitDatasetModule
from models import PCN
from metrics.metric import l1_cd, l2_cd, f_score
from metrics.precision_reall import PrecisionRecall
from pytorch3d.structures import Meshes

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def test(params, save=False):
    if save:
        make_dir(params.result_dir)

    print(params.exp_name)

    # load pretrained model
    # model = PCN(params, params.fine_p, 2024, 4).to(params.device)
    model = PCN(params, params.fine_p, 2048, 4).to(params.device)

    model.load_state_dict(torch.load(params.ckpt_path))
    model.eval()
    if save:
        cat_dir = os.path.join(params.result_dir)
        image_dir = os.path.join(cat_dir, 'image')
        output_dir = os.path.join(cat_dir, 'output')
        make_dir(cat_dir)
        make_dir(image_dir)
        make_dir(output_dir)

    data = IGGFruitDatasetModule(params)
    # val_dataloader = data.val_dataloader()

    test_dataloader = data.test_dataloader()

    index = 1
    total_l1_cd, total_l2_cd, total_f_score = 0.0, 0.0, 0.0
    precision_recall = PrecisionRecall(
        min_t=0.001, max_t=0.01, num=100)
    with torch.no_grad():
        for item in test_dataloader:
            # if "lab" in gt_path[0]:
            #     continue
            # print(item['fid'])
            p = item['p'].to(params.device)
            _, pt_pcd = model(p)

            # a = Meshes(
            #     verts=pt_pcd, faces=model.template2_faces
            # )
            # o3d.io.write_triangle_mesh("test1.obj", a)

            pt_pcd = pt_pcd.squeeze()
            # pt_mesh = o3d.geometry.TriangleMesh()
            # pt_mesh.vertices = o3d.utility.Vector3dVector(pt_pcd.cpu())
            # pt_mesh.triangles = o3d.utility.Vector3iVector(
            #     model.template2_faces[0].cpu())
            pt = pt_pcd.detach().cpu().numpy()
            pt_pcd = o3d.geometry.PointCloud()
            pt_pcd.points = o3d.utility.Vector3dVector(pt[:, :3])

            # o3d.io.write_point_cloud("/data2/uqzche20/shape_completion_challenge/val_prediction_coarse/"+item['fid'][0]+".ply", pt_pcd)




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, default="pepper", help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, default="/checkpoints/best_006_f_77.53_p_67.53_r_96.16.pth", help='The path of pretrained model.')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=8, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    parser.add_argument('--emd', type=bool, default=False, help='Whether evaluate emd')
    parser.add_argument('--PATH', type=str, default='/data2/uqzche20/shape_completion_challenge/', help='Model saving frequency')
    parser.add_argument('--inference', type=bool, default=True, help='Whether evaluate emd')
    parser.add_argument('--fine_d', type=float, default=7, help='Model saving frequency')

    params = parser.parse_args()
    d_p_dic = {7: 163840, 6: 40960, 5: 10240, 4: 2560, 3: 640, 2: 160}
    params.coarse_d = params.fine_d - 2

    params.coarse_p = d_p_dic[params.coarse_d]
    params.fine_p = d_p_dic[params.fine_d]

    test(params, params.save)
