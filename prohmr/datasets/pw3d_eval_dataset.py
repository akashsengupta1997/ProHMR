import cv2
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


class PW3DEvalDataset(Dataset):
    def __init__(self,
                 pw3d_dir_path,
                 img_wh=224,
                 visible_joints_threshold=0.6,
                 selected_fnames=None):
        super(PW3DEvalDataset, self).__init__()

        # Paths
        self.cropped_frames_dir = os.path.join(pw3d_dir_path, 'cropped_frames')
        self.hrnet_kps_dir = os.path.join(pw3d_dir_path, 'hrnet_results_centred', 'keypoints')

        # Data
        data = np.load(os.path.join(pw3d_dir_path, '3dpw_test.npz'))
        self.frame_fnames = data['imgname']
        self.pose = data['pose']
        self.shape = data['shape']
        self.gender = data['gender']

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

        self.img_wh = img_wh
        self.visible_joints_threshold = visible_joints_threshold
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

        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        orig_width, orig_height = img.shape[:2]
        img = cv2.resize(img, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1])/255.0

        # Targets
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]

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
                'vis_img': img,
                'pose': pose,
                'shape': shape,
                'hrnet_kps': hrnet_kps,
                'hrnet_kps_vis': hrnet_kps_vis_flag,
                'fname': fname,
                'gender': gender}








