from torch.utils.data import DataLoader, Dataset

import numpy as np
import open3d as o3d
import json
import os
import cv2
import random
import math
class IGGFruit(Dataset):

    def __init__(self,
                 data_source=None,
                 split='train',
                 sensor='realsense',
                 overfit=False,
                 precomputed_augmentation=False):
        super().__init__()

        # assert sensor in ['laser', 'realsense'], 'sensor {} not recognized'

        self.data_source = data_source
        self.sensor = sensor
        self.split = split
        self.overfit = overfit
        self.return_pcd = True
        self.fruit_list = self.get_file_paths()


    def get_file_paths(self):
        fruit_list = {}
        for fid in os.listdir(os.path.join(self.data_source, self.split)):
            fruit_list[fid] = {
                'path': os.path.join(self.data_source, self.split, fid),
            }
        return fruit_list

    def get_gt(self, fid):
        return o3d.io.read_point_cloud(os.path.join(self.fruit_list[fid]['path'], 'gt/pcd/fruit.ply'))

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
        # nframes = len(frames)
        # nlower = math.ceil(nframes*0.3)
        # nsample = random.randrange(nlower, nframes+1)
        # frames = random.sample(frames, k=nsample)
        for frameid in frames:

            pose_path = os.path.join(fid_root, 'input/poses/', frameid.replace('png', 'txt'))
            pose = np.loadtxt(pose_path)

            rgb_path = os.path.join(fid_root, 'input/color/', frameid)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

            depth_path = os.path.join(fid_root, 'input/depth/', frameid.replace('png', 'npy'))
            depth = np.load(depth_path)

            mask_path = os.path.join(fid_root, 'input/masks/', frameid)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            frame_key = frameid.replace('.png', '')

            frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
            # rgbd_data['pcd'] += frame_pcd
            bbox = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[0], max_bound=bbox[1])
            frame_pcd = frame_pcd.crop(bbox)

            o3d.io.write_point_cloud(self.fruit_list[fid]['path']+"/incomplete"+frame_key+".ply", frame_pcd)
            print("wrote to", self.fruit_list[fid]['path']+"/incomplete"+frame_key+".ply")
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

    @staticmethod
    def collate(batch):
        item = {}
        for key in batch[0].keys():
            item[key] = []

        for key in item.keys():
            for data in batch:
                item[key].append(data[key])

        return item

    def __getitem__(self, idx):
        # idx = 0
        keys = list(self.fruit_list.keys())
        fid = keys[idx]

        gt_pcd = self.get_gt(fid)
        input_data = self.get_rgbd(fid)
        item = {
            'gt_points': np.asarray(gt_pcd.points),
            'rgbd_pcd':np.asarray(input_data['pcd'].points),
            'rgbd_colors':np.asarray(input_data['pcd'].colors)
        }

        return item

    def process(self):
        for i in range(len(self.fruit_list)):

            idx = i
            keys = list(self.fruit_list.keys())
            fid = keys[idx]

            # gt_pcd = self.get_gt(fid)
            input_data = self.get_rgbd(fid)

            # item = {
            #     'gt_points': np.asarray(gt_pcd.points),
            #     'rgbd_pcd': np.asarray(input_data['pcd'].points),
            #     'rgbd_colors': np.asarray(input_data['pcd'].colors)
            # }

source = '/data2/uqzche20/shape_completion_challenge/'

# we construct partial observation for each frame and druing training we random choose some frames as the input
traindata = IGGFruit(source, split='train')
traindata.process()


