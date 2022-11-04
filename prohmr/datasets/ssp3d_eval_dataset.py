import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from prohmr.datasets.utils import batch_crop_opencv_affine


class SSP3DEvalDataset(Dataset):
    def __init__(self,
                 ssp3d_dir_path,
                 img_wh=224,
                 bbox_scale_factor=1.2,
                 visible_joints_threshold=None,
                 selected_fnames=None,
                 vis_img_wh=512,
                 extreme_crop=False,
                 extreme_crop_scale=None):
        super(SSP3DEvalDataset, self).__init__()

        # Paths
        self.images_dir = os.path.join(ssp3d_dir_path, 'images')
        self.pointrend_masks_dir = os.path.join(ssp3d_dir_path, 'silhouettes')

        # Data
        data = np.load(os.path.join(ssp3d_dir_path, 'labels.npz'))

        self.frame_fnames = data['fnames']
        self.body_shapes = data['shapes']
        self.body_poses = data['poses']
        self.kprcnn_kps = data['joints2D']
        self.bbox_centres = data['bbox_centres']  # Tight bounding box centre
        self.bbox_whs = data['bbox_whs']  # Tight bounding box width/height
        self.genders = data['genders']

        if selected_fnames is not None:  # Evaluate only given fnames
            chosen_indices = []
            for fname in selected_fnames:
                chosen_indices.append(np.where(self.frame_fnames == fname)[0])
                print(fname, np.where(self.frame_fnames == fname)[0])
            chosen_indices = np.concatenate(chosen_indices, axis=0)
            self.frame_fnames = self.frame_fnames[chosen_indices]
            self.body_poses = self.body_poses[chosen_indices]
            self.body_shapes = self.body_shapes[chosen_indices]
            self.kprcnn_kps = self.kprcnn_kps[chosen_indices]
            self.bbox_centres = self.bbox_centres[chosen_indices]
            self.bbox_whs = self.bbox_whs[chosen_indices]
            self.genders = self.genders[chosen_indices]

        assert len(self.frame_fnames) == len(self.body_shapes) == len(self.kprcnn_kps) == len(self.genders)

        self.img_wh = img_wh
        self.bbox_scale_factor = bbox_scale_factor
        self.visible_joints_threshold = visible_joints_threshold
        self.vis_img_wh = vis_img_wh

        self.extreme_crop = extreme_crop
        self.extreme_crop_scale = extreme_crop_scale

        self.normalize_img = Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # ------------------- Inputs -------------------
        fname = self.frame_fnames[index]
        image_path = os.path.join(self.images_dir, fname)

        # Frames + Joints need to be cropped to bounding box.
        # (Ideally should do this in pre-processing and store cropped frames + joints in npz files?)
        bbox_centre = self.bbox_centres[index]
        bbox_wh = self.bbox_whs[index]

        if self.extreme_crop:
            if self.extreme_crop_scale is None:
                bbox_centre[0] -= (1 - 0.5) * (bbox_wh / 2.0)
                bbox_wh *= (0.5 - 0.05)  # What's the 0.05 term here?
            else:
                bbox_centre[0] -= (1 - self.extreme_crop_scale) * (bbox_wh / 2.0)
                bbox_wh *= (self.extreme_crop_scale - 0.05)  # What's the 0.05 term here?

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img = batch_crop_opencv_affine(output_wh=(self.img_wh, self.img_wh),
                                       num_to_crop=1,
                                       rgb=img[None].transpose(0, 3, 1, 2),
                                       bbox_centres=bbox_centre[None],
                                       bbox_whs=[bbox_wh],
                                       orig_scale_factor=self.bbox_scale_factor)['rgb'][0] / 255.0

        vis_img = cv2.resize(np.transpose(img, [1, 2, 0]),
                             (self.vis_img_wh, self.vis_img_wh), interpolation=cv2.INTER_LINEAR)

        # ------------------- Targets -------------------
        shape = self.body_shapes[index]
        pose = self.body_poses[index]
        gender = self.genders[index]

        silhouette = cv2.imread(os.path.join(self.pointrend_masks_dir, fname), 0)
        silhouette = batch_crop_opencv_affine(output_wh=(self.img_wh, self.img_wh),
                                              num_to_crop=1,
                                              seg=silhouette[None],
                                              bbox_centres=bbox_centre[None],
                                              bbox_whs=[bbox_wh],
                                              orig_scale_factor=self.bbox_scale_factor)['seg'][0]

        kprcnn_kps = np.copy(self.kprcnn_kps[index])[:, :2]
        kprcnn_kps = batch_crop_opencv_affine(output_wh=(self.img_wh, self.img_wh),
                                              num_to_crop=1,
                                              joints2D=kprcnn_kps[None, :, :],
                                              bbox_centres=bbox_centre[None],
                                              bbox_whs=[bbox_wh],
                                              orig_scale_factor=self.bbox_scale_factor)['joints2D'][0]

        img = torch.from_numpy(img).float()
        shape = torch.from_numpy(shape).float()
        pose = torch.from_numpy(pose).float()

        input = self.normalize_img(img)

        return {'input': input,
                'vis_img': vis_img,
                'shape': shape,
                'pose': pose,
                'silhouette': silhouette,
                'keypoints': kprcnn_kps,
                'fname': fname,
                'gender': gender}
