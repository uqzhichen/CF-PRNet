import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import datetime
import random
from dataset.fruit import IGGFruitDatasetModule
import torch
torch.cuda.set_device('cuda:0')
import torch.optim as Optim
import ot
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
import open3d as o3d
from dataset import ShapeNet
from models import PCN
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1 #, emd_loss
from visualization import plot_pcd_one_view
from metrics.precision_reall import PrecisionRecall
import numpy as np

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, params.category)
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, params.category, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(params.exp_name), False)
    log(log_fd, "Logger directory: {}".format(logger_path), False)
    log(log_fd, str(params), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def train(params):
    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)

    log(log_fd, 'Loading Data...')

    data = IGGFruitDatasetModule(params)
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    # model
    model = PCN(params, num_dense=params.fine_p, latent_dim=params.latent_d, grid_size=4).to(params.device) #1024

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999))

    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    step = len(train_dataloader) // params.log_frequency

    # load pretrained model and optimizer
    if params.ckpt_path is not None:
        model.load_state_dict(torch.load(params.ckpt_path))

    # training
    best_f = 0
    best_epoch_f = -1
    train_step, val_step = 0, 0
    for epoch in range(1, params.epochs + 1):
        # hyperparameter alpha
        alpha = 1 # 0.2*epoch

        # training
        model.train()
        for i, (p, c) in enumerate(train_dataloader):
            p, c = p.to(params.device), c.to(params.device)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(p)

            loss1 = cd_loss_L1(coarse_pred, c)
            loss2 = cd_loss_L1(dense_pred, c)
            loss = loss1 + alpha * loss2
            deformed_template_mesh = Meshes(
                verts=dense_pred, faces=model.template2_faces
            )
            loss_normal = mesh_normal_consistency(deformed_template_mesh)
            loss_laplacian = mesh_laplacian_smoothing(deformed_template_mesh, method="uniform")
            # deformed_template_mesh = Meshes(
            #     verts=model.template2_points, faces=model.template2_faces
            # )
            # loss_normal += 10* mesh_normal_consistency(deformed_template_mesh)
            # loss_laplacian += 10* mesh_laplacian_smoothing(deformed_template_mesh, method="uniform")


            loss += params.normal_loss * loss_normal # eculidean
            loss += params.lap_loss * loss_laplacian
            # back propagation
            loss.backward()
            optimizer.step()

            if (i + 1) % step == 0:
                log(log_fd, "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: coarse loss = {:.6f}, "
                            "dense l1 cd = {:.6f}, normal = {:.6f}, lap = {:.6f}, total loss = {:.6f}"
                    .format(epoch, params.epochs, i + 1, len(train_dataloader), loss1.item(), loss2.item(),
                            loss_normal.item(), loss_laplacian.item(), loss.item()))
            
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
            train_writer.add_scalar('normal', loss_normal.item(), train_step)
            train_writer.add_scalar('laplacian', loss_laplacian.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
            train_step += 1
            torch.cuda.empty_cache()

        lr_schedual.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        precision_recall = PrecisionRecall(
            min_t=0.001, max_t=0.01, num=100)
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization

            for i, item in enumerate(val_dataloader):
                gt_path = item['gt']

                if "lab" in gt_path[0]:
                    continue

                p = item['p'].to(params.device)
                coarse, pt_pcd = model(p)

                pt_pcd = pt_pcd.squeeze()

                pt_pcd = pt_pcd.detach().cpu().numpy()
                pt = o3d.geometry.PointCloud()
                pt.points = o3d.utility.Vector3dVector(pt_pcd[:, :3])

                # pt_mesh = o3d.geometry.TriangleMesh()
                # pt_mesh.vertices = o3d.utility.Vector3dVector(pt_pcd.cpu())
                # pt_mesh.triangles = o3d.utility.Vector3iVector(
                #     model.template2_faces[0].cpu())


                gt_pcd_array = np.array(o3d.io.read_point_cloud(gt_path[0]).points)
                gt_pcd = o3d.geometry.PointCloud()
                gt_pcd.points = o3d.utility.Vector3dVector(gt_pcd_array[:, :3])

                precision_recall.update(gt_pcd, pt)
                if rand_iter == i:
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                                      [p[0].detach().cpu().numpy(), coarse[0].detach().cpu().numpy(), pt_pcd,
                                       item['gt_pcd'][0].detach().cpu().numpy()],
                                      ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))

            p, r, f = precision_recall.compute_auc()
            val_writer.add_scalar('f', f, val_step)
            val_writer.add_scalar('p', p, val_step)
            val_writer.add_scalar('r', r, val_step)

            val_step += 1

            log(log_fd, "Validate Epoch [{:03d}/{:03d}]: fscore= {:.6f}, precision={:.6f}, recall={:.6}".format(
                epoch, params.epochs, f, p, r))
            if f > best_f:
                best_epoch_f = epoch
                best_f = f
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_{:03d}_f_{:.2f}_p_{:.2f}_r_{:.2f}.pth'.format(
                epoch, f, p, r)))

    log(log_fd, 'Best f1 model in epoch {}, the maximum f1 is {}'.format(best_epoch_f, best_f))
    log_fd.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCN')
    parser.add_argument('--exp_name', type=str, default="pepper_test1", help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.00003, help='Learning rate') #0.0001 1e-4
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for data loader')
    parser.add_argument('--coarse_loss', type=str, default='cd', help='loss function for coarse point cloud')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--log_frequency', type=int, default=4, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')
    parser.add_argument('--PATH', type=str, default='/data2/uqzche20/shape_completion_challenge/', help='Model saving frequency')
    parser.add_argument('--inference', type=int, default=False, help='Model saving frequency')

    parser.add_argument('--normal_loss', type=float, default=0.0003, help='Model saving frequency')
    parser.add_argument('--lap_loss', type=float, default=0.001, help='Model saving frequency')
    parser.add_argument('--wd', type=float, default=3e-6, help='Weight Decay')
    parser.add_argument('--fine_d', type=float, default=7, help='Fine-grained template expansion degree')

    parser.add_argument('--latent_d', type=float, default=2048, help='Fine-grained template expansion degree')

    params = parser.parse_args()

    d_p_dic = {8: 655360, 7: 163840, 6: 40960, 5: 10240, 4: 2560, 3: 640, 2: 160}
    params.coarse_d = params.fine_d - 2

    params.coarse_p = d_p_dic[params.coarse_d]
    params.fine_p = d_p_dic[params.fine_d]

    train(params)
