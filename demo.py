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
import argparse
import os
import cv2
from tqdm import tqdm

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR
from prohmr.optimization import KeypointFitting
from prohmr.utils import recursive_to
from prohmr.datasets import OpenPoseDataset
from prohmr.utils.renderer import Renderer

parser = argparse.ArgumentParser(description='ProHMR demo code')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--img_folder', type=str, required=True, help='Folder with input images')
parser.add_argument('--keypoint_folder', type=str, required=True, help='Folder with corresponding OpenPose detections')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--out_format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format')
parser.add_argument('--run_fitting', dest='run_fitting', action='store_true', default=False, help='If set, run fitting on top of regression')
parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, run fitting in the original image space and not in the crop.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
parser.add_argument('--num_samples', type=int, default=25, help='Number of test samples to draw')


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

if args.run_fitting:
    keypoint_fitting = KeypointFitting(model_cfg)

# Create a dataset on-the-fly
dataset = OpenPoseDataset(model_cfg, img_folder=args.img_folder, keypoint_folder=args.keypoint_folder, max_people_per_image=1)

# Setup a dataloader with batch_size = 1 (Process images sequentially)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

# Go over each image in the dataset
for i, batch in enumerate(tqdm(dataloader)):

    batch = recursive_to(batch, device)
    with torch.no_grad():
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
        # print(out.keys())
        # for key in out:
        #     print('\n', key)
        #     if isinstance(out[key], torch.Tensor):
        #         print(out[key].shape)
        #     elif isinstance(out[key], dict):
        #         for key2 in out[key]:
        #             print(key2, out[key][key2].shape)

    batch_size = batch['img'].shape[0]
    for n in range(batch_size):
        img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
        regression_img = renderer(out['pred_vertices'][n, 0].detach().cpu().numpy(),
                                  out['pred_cam_t'][n, 0].detach().cpu().numpy(),
                                  batch['img'][n])
        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_regression.{args.out_format}'), 255*regression_img[:, :, ::-1])
    if args.run_fitting:
        opt_out = model.downstream_optimization(regression_output=out,
                                                batch=batch,
                                                opt_task=keypoint_fitting,
                                                use_hips=False,
                                                full_frame=args.full_frame)
        for n in range(batch_size):
            img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
            fitting_img = renderer(opt_out['vertices'][n].detach().cpu().numpy(),
                                   opt_out['camera_translation'][n].detach().cpu().numpy(),
                                   batch['img'][n], imgname=batch['imgname'][n], full_frame=args.full_frame)
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_fitting.{args.out_format}'), 255*fitting_img[:, :, ::-1])
