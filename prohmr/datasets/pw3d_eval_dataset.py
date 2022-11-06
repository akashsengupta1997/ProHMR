import cv2
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from prohmr.datasets.utils import batch_crop_opencv_affine


class PW3DEvalDataset(Dataset):
    def __init__(self,
                 pw3d_dir_path,
                 img_wh=224,
                 vis_img_wh=512,
                 visible_joints_threshold=0.6,
                 gt_visible_joints_threhshold=0.6,
                 selected_fnames=None,
                 extreme_crop=False,
                 extreme_crop_scale=None):
        super(PW3DEvalDataset, self).__init__()

        # Paths
        if not extreme_crop:
            self.cropped_frames_dir = os.path.join(pw3d_dir_path, 'cropped_frames')
            self.hrnet_kps_dir = os.path.join(pw3d_dir_path, 'hrnet_results_centred', 'keypoints')
        else:
            self.cropped_frames_dir = os.path.join(pw3d_dir_path, f'extreme_cropped_{extreme_crop_scale}_frames')
            self.hrnet_kps_dir = os.path.join(pw3d_dir_path, f'extreme_cropped_{extreme_crop_scale}_hrnet_results_centred', 'keypoints')
        print('3DPW Frames Dir:', self.cropped_frames_dir)
        print('3DPW HRNet KP Dir:', self.hrnet_kps_dir)

        # Data
        data = np.load(os.path.join(pw3d_dir_path, '3dpw_test2.npz'))
        self.frame_fnames = data['imgname']
        self.pose = data['pose']
        self.shape = data['shape']
        self.gender = data['gender']
        if not extreme_crop:
            self.joints2D_coco = data['joints2D_coco']
        else:
            joints2D_coco_path = os.path.join(pw3d_dir_path, f'extreme_cropped_{extreme_crop_scale}_joints2D_coco.npy')
            print('3DPW GT KP Dir:', joints2D_coco_path)
            self.joints2D_coco = np.load(joints2D_coco_path)

        if selected_fnames is not None:  # Evaluate only given fnames
            chosen_indices = []
            for fname in selected_fnames:
                chosen_indices.append(np.where(self.frame_fnames == fname)[0])
                print(fname, np.where(self.frame_fnames == fname)[0])
            chosen_indices = np.concatenate(chosen_indices, axis=0)
            self.frame_fnames = self.frame_fnames[chosen_indices]
            self.pose = self.pose[chosen_indices]
            self.shape = self.shape[chosen_indices]
            self.gender = self.gender[chosen_indices]
            self.joints2D_coco = self.joints2D_coco[chosen_indices]

        self.img_wh = img_wh
        self.vis_img_wh = vis_img_wh
        self.visible_joints_threshold = visible_joints_threshold
        self.gt_visible_joints_threhshold = gt_visible_joints_threhshold
        self.normalize_img = Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Inputs
        fname = self.frame_fnames[index]
        frame_path = os.path.join(self.cropped_frames_dir, fname)

        orig_img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        orig_height, orig_width = orig_img.shape[:2]
        img = cv2.resize(orig_img, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1]) / 255.0

        vis_img = cv2.resize(np.transpose(img, [1, 2, 0]),
                             (self.vis_img_wh, self.vis_img_wh), interpolation=cv2.INTER_LINEAR)

        # Targets
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]

        joints2D_coco = self.joints2D_coco[index]  # (17, 3) GT 2D joints or keypoints
        joints2D_coco_conf = joints2D_coco[:, 2]
        joints2D_coco = joints2D_coco[:, :2] * np.array([self.img_wh / float(orig_width),
                                                         self.img_wh / float(orig_height)])
        joints2D_coco_vis = joints2D_coco_conf > self.gt_visible_joints_threhshold
        joints2D_coco_vis[[1, 2, 3, 4]] = joints2D_coco_conf[[1, 2, 3, 4]] > 0.1  # Different threshold for these because confidences are generally very low for GT 2D keypoints.

        hrnet_kps_path = os.path.join(self.hrnet_kps_dir, os.path.splitext(fname)[0] + '.npy')
        hrnet_kps = np.load(hrnet_kps_path)
        hrnet_kps_confidence = hrnet_kps[:, 2]  # (17,)
        hrnet_kps = hrnet_kps[:, :2]
        hrnet_kps = hrnet_kps * np.array([self.img_wh / float(orig_width),
                                          self.img_wh / float(orig_height)])
        hrnet_kps_vis_flag = hrnet_kps_confidence > self.visible_joints_threshold
        hrnet_kps_vis_flag[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded

        img = torch.from_numpy(img).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        input = self.normalize_img(img)

        return {'input': input,
                'vis_img': vis_img,
                'pose': pose,
                'shape': shape,
                'hrnet_kps': hrnet_kps,
                'hrnet_kps_vis': hrnet_kps_vis_flag,
                'gt_kps': joints2D_coco,
                'gt_kps_vis': joints2D_coco_vis,
                'fname': fname,
                'gender': gender}








