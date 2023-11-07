import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from .base import BaseDataset
from .utils import to_numpy
from .builder import DATASETS
from mmdet.datasets.builder import PIPELINES
import os
import yaml
import numpy as np
from .pcd_transforms import *
from PIL import Image
# import open3d
import random


def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask


def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask

def get_polar_mask(pc, lims):
    r = np.sqrt(pc[:,0]**2 + pc[:,1]**2)
    mask_x = mask_op(r, lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & mask_z
    return mask


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def rotate_pc(pc, theta):
    rotation_matrix = np.array([np.cos([theta]),
                                np.sin([theta]),
                                -np.sin([theta]),
                                np.cos([theta])], dtype=np.float).reshape(2, 2)

    rxy = np.matmul(pc, rotation_matrix)

    return rxy


def random_rotate_pc(pc, xlim, ylim, theta):
    center_x = (xlim[0] + xlim[1]) / 2.
    center_y = (ylim[0] + ylim[1]) / 2.
    center = np.array([center_x, center_y], dtype=np.float)
    x = pc[:, 0] - center_x
    y = pc[:, 1] - center_y

    xy = np.stack([x, y], axis=-1)
    rxy = rotate_pc(xy, theta)

    add = rxy + center

    return add


def augmentation_rotate_pc(pc, lims):
    # angle = np.random.uniform() * np.pi * 2 #- np.pi
    angle = (-np.pi - np.pi) * torch.rand(1).tolist()[0] + np.pi
    rxy = random_rotate_pc(pc, lims[0], lims[1], angle)
    # print(pc[:, 0].size(), rxy[0].size())
    pc[:, :2] = rxy

    return pc


def augmentation_random_flip(pc):
    flip_type = torch.randint(4, (1,)).tolist()[0]
    if flip_type==1:
        pc[:,0] = -pc[:,0]
    elif flip_type==2:
        pc[:,1] = -pc[:,1]
    elif flip_type==3:
        pc[:,:2] = -pc[:,:2]
    return pc

def augmentation_scale(pc):
    noise_scale = np.random.uniform(0.95, 1.05)
    noise_scale = (0.95 - 1.05) * torch.rand(1).tolist()[0] + 1.05
    pc[:, 0] = noise_scale * pc[:, 0]
    pc[:, 1] = noise_scale * pc[:, 1]
    pc[:, 2] = noise_scale * pc[:, 2]
    # drop = prob
    return pc


def augmentation_rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    out = np.dot(batch_data[:, :3].reshape((-1, 3)), R)
    batch_data[:, :3] = out
    return batch_data

def random_jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, 3), -1*clip, clip)
    # jittered_data += batch_data
    pc[:, :3] += jittered_data
    return pc


def transforms(points):
    points = np.expand_dims(points, axis=0)
    points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
    points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
    points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
    points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
    points[:,:,:3] = jitter_point_cloud(points[:,:,:3])
    points = random_drop_n_cuboids(points)

    return np.squeeze(points, axis=0)

@DATASETS.register_module
class SemanticKitti(BaseDataset):
    CLASSES = ('unlabeled',
               'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
               'person', 'bicyclist', 'motorcyclist', 'road',
               'parking', 'sidewalk', 'other-ground', 'building', 'fence',
               'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign')

    def __init__(self, data_root, data_config_file, setname,
                 lims,
                 pipelines,
                 augmentation=False,
                 max_num=140000,
                 with_gt=False,
                 ignore_class=[0],
                 shuffle_index=False,
                 test_mode=False,
                 prefetch=False):
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipelines]
        print(setname, augmentation)
        self.n_clusters = 50
        self.prefetch = prefetch
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.data_config = yaml.safe_load(open(data_config_file, 'r'))
        self.sequences = self.data_config["split"][setname]
        self.setname = setname
        self.labels = self.data_config['labels']
        self.learning_map = self.data_config["learning_map"]
        print(self.learning_map)
        self.learning_map_inv = self.data_config["learning_map_inv"]
        self.with_gt = with_gt
        self.color_map = self.data_config['color_map']

        self.lims = lims
        self.augmentation = augmentation
        self.scan_files = []
        self.label_files = []
        self.shuffle_index = shuffle_index
        self.ignore_class = ignore_class
        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.data_root, seq, "velodyne")
            label_path = os.path.join(self.data_root, seq, "labels")

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]

            # check all scans have labels
            if self.with_gt:
                assert (len(scan_files) == len(label_files))

            # extend list
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        self.scan_files.sort()
        self.label_files.sort()
        self.file_idx = np.arange(0, len(self.scan_files))
        self.num_files_ = len(self.file_idx)
        print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))
        print(self.augmentation, " is aug")

        self._set_group_flag()
        
    
    def __len__(self):
        return self.num_files_

    def get_n_classes(self):
        return len(self.learning_map_inv)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def __getitem__(self, idx):
        data_dict = self.get_normal_item(idx)
        cut_mix_dict= {}
        if self.setname == 'train':
            cut_index = random.randint(0, self.__len__() - 1)

            while cut_index == idx:
                cut_index = random.randint(0, self.__len__() - 1)

            cut_dict = self.get_normal_item(cut_index, cut_scene = True)
            for keys in cut_dict.keys():
                if keys == 'points':
                    cut_mix_dict[keys] = torch.vstack((data_dict[keys], cut_dict[keys]))
                else:
                    cut_mix_dict[keys] = torch.cat((data_dict[keys], cut_dict[keys]), dim=0)
        else:
            cut_mix_dict = data_dict


        if self.with_gt:
            filter_label = cut_mix_dict['points_label']
            ###for sem maskfomer
            ### 0 is noise
            target_labels = np.unique(filter_label)[1:] #delete 0
            for idx,i in enumerate(target_labels):
                temp_target_points = np.zeros_like(filter_label)
                # temp = (filter_label==i).sum()
                temp_target_points[filter_label==i]=1
                if idx==0:
                    target_points=temp_target_points[np.newaxis,:]
                else :
                    target_points = np.concatenate((target_points,temp_target_points[np.newaxis,:]),axis=0)
            target_labels-= 1

            target_labels = torch.from_numpy(target_labels).long()
            target_points = torch.from_numpy(target_points)
            cut_mix_dict['target_labels'] = target_labels
            cut_mix_dict['target_masks'] = target_points

        example = self.pipeline(cut_mix_dict)
        return example

    def get_normal_item(self, idx, cut_scene=False):
        if idx >= self.num_files_:
            np.random.shuffle(self.file_idx)
        scan_file = self.scan_files[self.file_idx[idx]]
        scan = np.fromfile(scan_file, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        if self.with_gt:
            label_file = self.label_files[self.file_idx[idx]]
            label = np.fromfile(label_file, dtype=np.uint32)
            label = label.reshape((-1))
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
            assert ((sem_label + (inst_label << 16) == label).all())
            sem_label = self.map(sem_label, self.learning_map)
        if cut_scene:
            mask = inst_label != 0    
            scan = scan[mask]
            sem_label = sem_label[mask]
        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, scan.shape[0]))
            scan = scan[pt_idx]
            if self.with_gt:
                sem_label = sem_label[pt_idx]

        if self.augmentation:
            scan = augmentation_random_flip(scan)
            scan = augmentation_rotate_pc(scan, self.lims)
            scan = augmentation_scale(scan)
            scan = random_jitter_point_cloud(scan)
            scan = augmentation_rotate_perturbation_point_cloud(scan)


        if self.lims:
            filter_mask = get_mask(scan, self.lims)
            ori_num = filter_mask.shape[0]
            filter_scan = scan[filter_mask]
            if self.with_gt:
                filter_label = sem_label[filter_mask]
        else:
            filter_scan = scan
            if self.with_gt:
                filter_label = sem_label
            ori_num = scan.shape[0]
            filter_mask = np.ones(filter_scan.shape[0], np.bool)

        # if self.augmentation:
        #     filter_scan = augmentation_random_flip(filter_scan)
        #     filter_scan = augmentation_rotate_pc(filter_scan, self.lims)
        #     filter_scan = augmentation_scale(filter_scan)
        #     filter_scan = random_jitter_point_cloud(filter_scan)
        #     filter_scan = augmentation_rotate_perturbation_point_cloud(filter_scan)
        if self.with_gt:
            filter_label = torch.from_numpy(filter_label).int()
        scan_th = torch.as_tensor(filter_scan, dtype=torch.float32)
        filter_mask = torch.from_numpy(filter_mask).bool()

        if self.with_gt:
            input_dict = dict(points=scan_th, filter_mask=filter_mask, points_label=filter_label)
        else:
            input_dict = dict(points=scan_th, filter_mask=filter_mask)
            
        return input_dict
    def evaluate(self, results, logger=None):
        return NotImplemented
    
    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out
    
    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind