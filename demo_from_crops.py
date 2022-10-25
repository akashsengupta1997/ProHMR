"""
ProHMR demo script.
To run our method you need a folder with images and corresponding OpenPose detections.
These are used to crop the images around the humans and optionally to fit the SMPL model on the detections.

Example usage:
python demo.py --checkpoint=path/to/checkpoint.pt --img_folder=/path/to/images --keypoint_folder=/path/to/json --out_folder=/path/to/output --run_fitting

Running the above will run inference for all images in /path/to/images with corresponding keypoint detections.
The rendered results will be saved to /path/to/output, with the suffix _regression.jpg for the regression (mode) and _fitting.jpg for the fitting.

Please keep in mind that we do not recommend to use `--full_frame` when the image resolution is above 2K because of known issues with the data term of SMPLify.
In these cases you can resize all images such that the maximum image dimension is at most 2K.
"""
import torch
import numpy as np
from torchvision.transforms import Normalize
import argparse
import os
import cv2
from tqdm import tqdm

from prohmr.configs import get_config, prohmr_config
from prohmr.models import ProHMR
from prohmr.utils import recursive_to
from prohmr.utils.imutils_from_spin import crop
from prohmr.utils.renderer import Renderer

parser = argparse.ArgumentParser(description='ProHMR demo code')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--img_folder', type=str, required=True, help='Folder with input images')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--out_format', type=str, default='png', choices=['jpg', 'png'], help='Output image format')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
parser.add_argument('--num_samples', type=int, default=25, help='Number of test samples to draw')


def process_image(img_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    img = cv2.imread(img_file)[:, :, ::-1].copy()  # PyTorch does not support negative stride at the moment
    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200

    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

args = parser.parse_args()

# Use the GPU if available
gpu = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

# Setup model
model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()
model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples + 1
model_cfg.freeze()

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

# Go over each image in the image directory
img_fnames = sorted([f for f in os.listdir(args.img_folder) if f.endswith('.png') or f.endswith('.jpg')])
print('Image found:', len(img_fnames))

for i, fname in enumerate(tqdm(img_fnames)):
    img, norm_img = process_image(os.path.join(args.img_folder, fname), input_res=model_cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        "Need this stuff for actnorm init"
        smpl_params = {'global_orient': torch.zeros(3, dtype=torch.float32, device=device),
                       'body_pose': torch.zeros(69, dtype=torch.float32, device=device),
                       'betas': torch.zeros(10, dtype=torch.float32, device=device)
                       }
        has_smpl_params = {'global_orient': 0.,
                           'body_pose': 0.,
                           'betas': 0.
                           }
        batch = {'img': norm_img.to(device),
                 'smpl_params': smpl_params,
                 'has_smpl_params': has_smpl_params}
        out = model(batch)
        """
        out is a dict with keys:
        - pred_cam: (1, num_samples, 3) tensor, camera is same for all samples
        - pred_cam_t: (1, num_samples, 3) tensor, camera is same for all samples
        
        - pred_smpl_params: dict with keys:
                - global_orient: (1, num_samples, 1, 3, 3) tensor
                - body_pose: (1, num_samples, 23, 3, 3) tensor
                - betas: (1, num_samples, 10) tensor, betas are same for all samples
        - pred_pose_6d: (1, num_samples, 144) tensor
        - pred_vertices: (1, num_samples, 6890, 3) tensor
        - pred_keypoints_3d: (1, num_samples, 44, 3) tensor
        - pred_keypoints_2d: (1, num_samples, 44, 2) tensor
        
        - log_prob: (1, num_samples) tensor
        - conditioning_feats: (1, 2047) tensor
        """

    batch_size = batch['img'].shape[0]
    for n in range(batch_size):
        regression_img = renderer(out['pred_vertices'][n, 0].detach().cpu().numpy(),
                                  out['pred_cam_t'][n, 0].detach().cpu().numpy(),
                                  batch['img'][n])
        cv2.imwrite(os.path.join(args.out_folder, f'{os.path.splitext(fname)[0]}_regression.{args.out_format}'),
                    255*regression_img[:, :, ::-1])
