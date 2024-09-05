from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import open3d as o3d
import json
import os
import cv2
import math
import random

class IGGFruit(Dataset):

    def __init__(self,
                cfg,
                 data_source=None,
                 split='train',
                 sensor='realsense',
                 overfit=False,
                 precomputed_augmentation=False,
                 inference=False):
        super().__init__()
        self.cfg = cfg
        self.inference = inference
        # assert sensor in ['laser', 'realsense'], 'sensor {} not recognized'

        self.data_source = data_source
        self.split = split
        self.overfit = overfit

        self.return_pcd = True

        self.fruit_list = self.get_file_paths()


    def get_file_paths(self):
        fruit_list = {}
        for fid in os.listdir(os.path.join(self.data_source, self.split)):
            # if self.split!="train" and not fid.startswith("p"): #
            #     continue
            fruit_list[fid] = {
                'path': os.path.join(self.data_source, self.split, fid),
            }
        return fruit_list

    def get_gt(self, fid):
        return o3d.io.read_point_cloud(os.path.join(self.fruit_list[fid]['path'], 'gt/pcd/fruit.ply'))

    def get_processed_input(self, fid):
        # print(fid)
        if self.split in ["val", "test"]:
            return o3d.io.read_point_cloud(os.path.join(self.fruit_list[fid]['path'], 'incomplete.ply'))

        frames = [frame
                  for frame in os.listdir(os.path.join(self.data_source, self.split+"/"+fid+'/'))
                  if frame.startswith("incomplete")]

        nframes = len(frames)
        nlower = math.ceil(nframes*0.3)
        nsample = random.randrange(nlower, nframes+1)
        frames = random.sample(frames, k=nsample)

        pcd = o3d.geometry.PointCloud()
        for frame in frames:
            frame = o3d.io.read_point_cloud(self.fruit_list[fid]['path'] + "/" + frame)
            pcd += frame

        if len(pcd.points) > 1e+5:
            for i in np.arange(0.0005,0.003, 0.0001):
                pcd_temp = pcd.voxel_down_sample(voxel_size=i)
                # print(len(pcd.points))
                if len(pcd_temp.points) < 1e+5:
                    pcd = pcd_temp
                    break

        return pcd

    def get_rgbd(self, fid):
        fid_root = self.fruit_list[fid]['path']

        intrinsic_path = os.path.join(fid_root, 'input/intrinsic.json')
        intrinsic = self.load_K(intrinsic_path)

        rgbd_data = {
            'intrinsic': intrinsic,
            'pcd': o3d.geometry.PointCloud(),
            'frames': {}
        }

        frames = os.listdir(os.path.join(fid_root, 'input/masks/'))
        for frameid in frames:

            pose_path = os.path.join(fid_root, 'input/poses/', frameid.replace('png', 'txt'))
            pose = np.loadtxt(pose_path)

            rgb_path = os.path.join(fid_root, 'input/color/', frameid)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

            depth_path = os.path.join(fid_root, 'input/depth/', frameid.replace('png', 'npy'))
            depth = np.load(depth_path)

            mask_path = os.path.join(fid_root, 'input/masks/', frameid)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            frame_key = frameid.replace('png', '')

            if self.return_pcd:
                frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
                rgbd_data['pcd'] += frame_pcd

            rgbd_data['frames'][frame_key] = {
                'rgb': rgb,
                'depth': depth,
                'mask': mask,
                'pose': pose
            }

        # o3d.io.write_point_cloud("toriginal.ply",  rgbd_data['pcd'])
        bbox = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[0], max_bound=bbox[1])
        rgbd_data['pcd'] = rgbd_data['pcd'].crop(bbox)

        rgbd_data['pcd'] = rgbd_data['pcd'].voxel_down_sample(voxel_size=0.0015)
        # o3d.io.write_point_cloud("tcrop.ply",  rgbd_data['pcd'])


        return rgbd_data

    @staticmethod
    def load_K(path):
        with open(path, 'r') as f:
            data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F')
        return k

    @staticmethod
    def rgbd_to_pcd(rgb, depth, mask, pose, K):

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),
                                                                  o3d.geometry.Image(depth * mask),
                                                                  depth_scale=1,
                                                                  depth_trunc=1.0,
                                                                  convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(height=rgb.shape[0],
                                 width=rgb.shape[1],
                                 fx=K[0, 0],
                                 fy=K[1, 1],
                                 cx=K[0, 2],
                                 cy=K[1, 2],
                                 )

        extrinsic = np.linalg.inv(pose)

        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
        return frame_pcd

    def __len__(self):
        return len(self.fruit_list)

    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]

    def __getitem__(self, idx):
        keys = list(self.fruit_list.keys())
        fid = keys[idx]


        input_data = self.get_processed_input(fid)
        input_data_points = np.array(input_data.points, np.float32)
        partial_pc = self.random_sample(input_data_points, self.cfg.coarse_p)

        if self.split == "test":
            return {"p": torch.from_numpy(partial_pc),
                    'fid': fid}

        gt_pcd = self.get_gt(fid)
        gt_pcd_points = np.array(gt_pcd.points, np.float32)
        complete_pc = self.random_sample(gt_pcd_points, self.cfg.fine_p)

        if self.split=='val':
            return {"p": torch.from_numpy(partial_pc),
                    "gt": os.path.join(self.fruit_list[fid]['path'], 'gt/pcd/fruit.ply'),
                    'gt_pcd': torch.from_numpy(complete_pc),
                    'fid': fid}
        else:
            return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)




class IGGFruitDatasetModule():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = IGGFruit(
            cfg=self.cfg,
            data_source=cfg.PATH,
            split='train',
            )
        self.test_dataset = IGGFruit(
            cfg=self.cfg,
            data_source=cfg.PATH,
            split='test',
            inference=cfg.inference
            )
        self.val_dataset = IGGFruit(
            cfg=self.cfg,
            data_source=cfg.PATH,
            split='val',
            inference=cfg.inference
            )

    def train_dataloader(self):

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=8,
        )
        return self.train_loader

    def val_dataloader(self):

        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )
        return self.val_loader

    def test_dataloader(self):

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )
        return self.test_loader
