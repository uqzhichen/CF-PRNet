import os
import argparse

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data
from dataset.fruit import IGGFruitDatasetModule
from models import CFPRNet
from metrics.metric import l1_cd, l2_cd, f_score
from metrics.precision_reall import PrecisionRecall

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def test(params, save=False):
    if save:
        make_dir(params.result_dir)

    print(params.exp_name)

    # load pretrained model
    model = CFPRNet(params, params.fine_p, params.latent_d, 4).to(params.device)
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
    val_dataloader = data.val_dataloader()

    index = 1
    total_l1_cd, total_l2_cd, total_f_score = 0.0, 0.0, 0.0
    precision_recall = PrecisionRecall(
        min_t=0.001, max_t=0.01, num=100)
    with torch.no_grad():
        for item in val_dataloader:
            gt_path = item['gt']
            # if "lab" in gt_path[0]:
            #     continue
            print(gt_path)
            p = item['p'].to(params.device)
            _, pt_pcd = model(p)

            pt_pcd = pt_pcd.squeeze()
            # pt_mesh = o3d.geometry.TriangleMesh()
            # pt_mesh.vertices = o3d.utility.Vector3dVector(pt_pcd.cpu())
            # pt_mesh.triangles = o3d.utility.Vector3iVector(
            #     model.template2_faces[0].cpu())
            pt = pt_pcd.detach().cpu().numpy()
            pt_pcd = o3d.geometry.PointCloud()
            pt_pcd.points = o3d.utility.Vector3dVector(pt[:, :3])

            # o3d.io.write_point_cloud("templte_cloud.ply", pt_pcd)

            # pt_pcd = pt_pcd.sample_points_uniformly(1000000)

            # geom = pt_pcd.detach().cpu().numpy()
            # geom_pcd = o3d.geometry.PointCloud()
            # geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
            # geom_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # radii = [0.005, 0.01, 0.02, 0.04]
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            #     geom_pcd,
            #     o3d.utility.DoubleVector(radii))
            # pt_pcd = mesh.sample_points_uniformly(1000000)


            # print(gt_path)
            gt_pcd_array = np.array(o3d.io.read_point_cloud(gt_path[0]).points)
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_pcd_array[:, :3])

            precision_recall.update(gt_pcd, pt_pcd) #72.0
        p, r, f = precision_recall.compute_auc()


        print("val_precision_auc", p)
        print("val_recall_auc",r)
        print("val_fscore_auc", f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, default="pepper", help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, default="/checkpoints/best_006_f_77.53_p_67.53_r_96.16.pth", help='The path of pretrained model.')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=6, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    parser.add_argument('--emd', type=bool, default=False, help='Whether evaluate emd')
    parser.add_argument('--PATH', type=str, default='/data2/uqzche20/shape_completion_challenge/', help='Model saving frequency')
    parser.add_argument('--inference', type=bool, default=True, help='Whether evaluate emd')
    parser.add_argument('--fine_d', type=float, default=7, help='Model saving frequency')
    parser.add_argument('--latent_d', type=float, default=2048, help='Fine-grained template expansion degree')

    params = parser.parse_args()

    d_p_dic = {7: 163840, 6: 40960, 5: 10240, 4: 2560, 3: 640, 2: 160}
    params.coarse_d = params.fine_d - 2

    params.coarse_p = d_p_dic[params.coarse_d]
    params.fine_p = d_p_dic[params.fine_d]

    test(params, params.save)
