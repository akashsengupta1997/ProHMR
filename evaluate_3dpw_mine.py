import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2
import math

import my_config

from prohmr.datasets.pw3d_eval_dataset import PW3DEvalDataset
from prohmr.configs import get_config, prohmr_config
from prohmr.models import ProHMR
from prohmr.models.smpl_mine import SMPL
from prohmr.utils.pose_utils import compute_similarity_transform_batch_numpy, scale_and_translation_transform_batch
from prohmr.utils.geometry import undo_keypoint_normalisation, orthographic_project_torch
from prohmr.utils.renderer import Renderer
from prohmr.utils.sampling_utils import compute_vertex_uncertainties_from_samples

import subsets


def evaluate_3dpw(model,
                  eval_dataset,
                  metrics_to_track,
                  device,
                  save_path,
                  num_pred_samples,
                  num_workers=4,
                  pin_memory=True,
                  vis_every_n_batches=1000,
                  num_samples_to_visualise=10,
                  renderer=None,
                  save_per_frame_uncertainty=True):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl_neutral = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1)
    smpl_male = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1, gender='male')
    smpl_female = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1, gender='female')
    smpl_neutral.to(device)
    smpl_male.to(device)
    smpl_female.to(device)

    metric_sums = {'num_datapoints': 0}
    per_frame_metrics = {}
    for metric in metrics_to_track:
        metric_sums[metric] = 0.
        per_frame_metrics[metric] = []

        if metric == 'joints3D_coco_invis_samples_dist_from_mean':
            metric_sums['num_invis_joints3Dsamples'] =  0

        elif metric == 'hrnet_joints2D_l2es':
            metric_sums['num_vis_hrnet_joints2D'] = 0

        elif metric == 'hrnet_joints2Dsamples_l2es':
            metric_sums['num_vis_hrnet_joints2Dsamples'] = 0


    fname_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []
    if save_per_frame_uncertainty:
        vertices_uncertainty_per_frame = []

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # if batch_num == 2:
        #     break
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input'].to(device)
        target_pose = samples_batch['pose'].to(device)
        target_shape = samples_batch['shape'].to(device)
        target_gender = samples_batch['gender'][0]
        hrnet_joints2D_coco = samples_batch['hrnet_kps'].cpu().detach().numpy()
        hrnet_joints2D_coco_vis = samples_batch['hrnet_kps_vis'].cpu().detach().numpy()
        fname = samples_batch['fname']

        if target_gender == 'm':
            target_smpl_output = smpl_male(body_pose=target_pose[:, 3:],
                                           global_orient=target_pose[:, :3],
                                           betas=target_shape)
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_smpl_output = smpl_female(body_pose=target_pose[:, 3:],
                                             global_orient=target_pose[:, :3],
                                             betas=target_shape)
            target_reposed_smpl_output = smpl_female(betas=target_shape)

        target_vertices = target_smpl_output.vertices
        target_joints_h36mlsp = target_smpl_output.joints[:, my_config.ALL_JOINTS_TO_H36M_MAP, :][:, my_config.H36M_TO_J14, :]
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # ------------------------------- PREDICTIONS -------------------------------
        out = model({'img': input})
        """
        out is a dict with keys:
        - pred_cam: (1, num_samples, 3) tensor, camera is same for all samples
        - pred_cam_t: (1, num_samples, 3) tensor, camera is same for all samples
            - This is just pred_cam converted from weak-perspective (i.e. [s, tx, ty]) to 
              full-perspective (i.e. [tx, ty, tz] and focal_length = 5000 --> this is basically just weak-perspective anyway)

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
        pred_cam_wp = out['pred_cam'][:, 0, :]

        pred_pose_rotmats_mode = out['pred_smpl_params']['body_pose'][:, 0, :, :, :]
        pred_glob_rotmat_mode = out['pred_smpl_params']['global_orient'][:, 0, :, :, :]
        pred_shape_mode = out['pred_smpl_params']['betas'][:, 0, :]

        pred_pose_rotmats_samples = out['pred_smpl_params']['body_pose'][0, 1:, :, :, :]
        pred_glob_rotmat_samples = out['pred_smpl_params']['global_orient'][0, 1:, :, :, :]
        pred_shape_samples = out['pred_smpl_params']['betas'][0, 1:, :]
        assert pred_pose_rotmats_samples.shape[0] == num_pred_samples

        pred_smpl_output_mode = smpl_neutral(body_pose=pred_pose_rotmats_mode,
                                             global_orient=pred_glob_rotmat_mode,
                                             betas=pred_shape_mode,
                                             pose2rot=False)
        pred_vertices_mode = pred_smpl_output_mode.vertices  # (1, 6890, 3)
        pred_joints_h36mlsp_mode = pred_smpl_output_mode.joints[:, my_config.ALL_JOINTS_TO_H36M_MAP, :][:, my_config.H36M_TO_J14, :]  # (1, 14, 3)
        pred_joints_coco_mode = pred_smpl_output_mode.joints[:, my_config.ALL_JOINTS_TO_COCO_MAP, :]  # (1, 17, 3)

        pred_vertices2D_mode = orthographic_project_torch(pred_vertices_mode, pred_cam_wp, scale_first=False)
        pred_vertices2D_mode = undo_keypoint_normalisation(pred_vertices2D_mode, input.shape[-1])
        pred_joints2D_coco_mode = orthographic_project_torch(pred_joints_coco_mode, pred_cam_wp)  # (1, 17, 2)
        pred_joints2D_coco_mode = undo_keypoint_normalisation(pred_joints2D_coco_mode, input.shape[-1])

        pred_reposed_vertices_mean = smpl_neutral(betas=pred_shape_mode).vertices  # (1, 6890, 3)

        pred_smpl_output_samples = smpl_neutral(body_pose=pred_pose_rotmats_samples,
                                                global_orient=pred_glob_rotmat_samples,
                                                betas=pred_shape_samples,
                                                pose2rot=False)
        pred_vertices_samples = pred_smpl_output_samples.vertices  # (num_pred_samples, 6890, 3)
        pred_joints_h36mlsp_samples = pred_smpl_output_samples.joints[:, my_config.ALL_JOINTS_TO_H36M_MAP, :][:, my_config.H36M_TO_J14, :]  # (num_samples, 14, 3)

        pred_joints_coco_samples = pred_smpl_output_samples.joints[:, my_config.ALL_JOINTS_TO_COCO_MAP, :]  # (num_pred_samples, 17, 3)
        pred_joints2D_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)  # (num_pred_samples, 17, 2)
        pred_joints2D_coco_samples = undo_keypoint_normalisation(pred_joints2D_coco_samples, input.shape[-1])

        pred_reposed_vertices_samples = smpl_neutral(body_pose=torch.zeros(num_pred_samples, 69, device=device, dtype=torch.float32),
                                                     global_orient=torch.zeros(num_pred_samples, 3, device=device, dtype=torch.float32),
                                                     betas=pred_shape_samples).vertices  # (num_pred_samples, 6890, 3)

        # ------------------------------- METRICS -------------------------------
        # Numpy-fying targets
        target_vertices = target_vertices.cpu().detach().numpy()
        target_joints_h36mlsp = target_joints_h36mlsp.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

        # Numpy-fying preds
        pred_vertices_mode = pred_vertices_mode.cpu().detach().numpy()
        pred_joints_h36mlsp_mode = pred_joints_h36mlsp_mode.cpu().detach().numpy()
        pred_joints_coco_mode = pred_joints_coco_mode.cpu().detach().numpy()
        pred_vertices2D_mode = pred_vertices2D_mode.cpu().detach().numpy()
        pred_joints2D_coco_mode = pred_joints2D_coco_mode.cpu().detach().numpy()
        pred_reposed_vertices_mean = pred_reposed_vertices_mean.cpu().detach().numpy()

        pred_vertices_samples = pred_vertices_samples.cpu().detach().numpy()
        pred_joints_h36mlsp_samples = pred_joints_h36mlsp_samples.cpu().detach().numpy()
        pred_joints_coco_samples = pred_joints_coco_samples.cpu().detach().numpy()
        pred_joints2D_coco_samples = pred_joints2D_coco_samples.cpu().detach().numpy()
        pred_reposed_vertices_samples = pred_reposed_vertices_samples.cpu().detach().numpy()

        # -------------- 3D Metrics with Mode and Minimum Error Samples --------------
        if 'pves' in metrics_to_track:
            pve_batch = np.linalg.norm(pred_vertices_mode - target_vertices,
                                       axis=-1)  # (bs, 6890)
            metric_sums['pves'] += np.sum(pve_batch)  # scalar
            per_frame_metrics['pves'].append(np.mean(pve_batch, axis=-1))

        if 'pves_samples_min' in metrics_to_track:
            pve_per_sample = np.linalg.norm(pred_vertices_samples - target_vertices, axis=-1)  # (num samples, 6890)
            min_pve_sample = np.argmin(np.mean(pve_per_sample, axis=-1))
            pve_samples_min_batch = pve_per_sample[min_pve_sample]  # (6890,)
            metric_sums['pves_samples_min'] += np.sum(pve_samples_min_batch)
            per_frame_metrics['pves_samples_min'].append(np.mean(pve_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'pves_sc' in metrics_to_track:
            pred_vertices_sc = scale_and_translation_transform_batch(
                pred_vertices_mode,
                target_vertices)
            pve_sc_batch = np.linalg.norm(
                pred_vertices_sc - target_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pves_sc'] += np.sum(pve_sc_batch)  # scalar
            per_frame_metrics['pves_sc'].append(np.mean(pve_sc_batch, axis=-1))

        if 'pves_sc_samples_min' in metrics_to_track:
            target_vertices_tiled = np.tile(target_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_sc = scale_and_translation_transform_batch(
                pred_vertices_samples,
                target_vertices_tiled)
            pve_sc_per_sample = np.linalg.norm(pred_vertices_samples_sc - target_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pve_sc_sample = np.argmin(np.mean(pve_sc_per_sample, axis=-1))
            pve_sc_samples_min_batch = pve_sc_per_sample[min_pve_sc_sample]  # (6890,)
            metric_sums['pves_sc_samples_min'] += np.sum(pve_sc_samples_min_batch)
            per_frame_metrics['pves_sc_samples_min'].append(np.mean(pve_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Procrustes analysis
        if 'pves_pa' in metrics_to_track:
            pred_vertices_pa = compute_similarity_transform_batch_numpy(pred_vertices_mode, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bs, 6890)
            metric_sums['pves_pa'] += np.sum(pve_pa_batch)  # scalar
            per_frame_metrics['pves_pa'].append(np.mean(pve_pa_batch, axis=-1))

        if 'pves_pa_samples_min' in metrics_to_track:
            target_vertices_tiled = np.tile(target_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_pa = compute_similarity_transform_batch_numpy(
                pred_vertices_samples,
                target_vertices_tiled)
            pve_pa_per_sample = np.linalg.norm(pred_vertices_samples_pa - target_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pve_pa_sample = np.argmin(np.mean(pve_pa_per_sample, axis=-1))
            pve_pa_samples_min_batch = pve_pa_per_sample[min_pve_pa_sample]  # (6890,)
            metric_sums['pves_pa_samples_min'] += np.sum(pve_pa_samples_min_batch)
            per_frame_metrics['pves_pa_samples_min'].append(np.mean(pve_pa_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        if 'pve-ts' in metrics_to_track:
            pvet_batch = np.linalg.norm(pred_reposed_vertices_mean - target_reposed_vertices, axis=-1)
            metric_sums['pve-ts'] += np.sum(pvet_batch)  # scalar
            per_frame_metrics['pve-ts'].append(np.mean(pvet_batch, axis=-1))

        if 'pve-ts_samples_min' in metrics_to_track:
            pvet_per_sample = np.linalg.norm(pred_reposed_vertices_samples - target_reposed_vertices, axis=-1)  # (num samples, 6890)
            min_pvet_sample = np.argmin(np.mean(pvet_per_sample, axis=-1))
            pvet_samples_min_batch = pvet_per_sample[min_pvet_sample]  # (6890,)
            metric_sums['pve-ts_samples_min'] += np.sum(pvet_samples_min_batch)
            per_frame_metrics['pve-ts_samples_min'].append(np.mean(pvet_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'pve-ts_sc' in metrics_to_track:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices_mean,
                target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(
                pred_reposed_vertices_sc - target_reposed_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pve-ts_sc'] += np.sum(pvet_scale_corrected_batch)  # scalar
            per_frame_metrics['pve-ts_sc'].append(np.mean(pvet_scale_corrected_batch, axis=-1))

        if 'pve-ts_sc_samples_min' in metrics_to_track:
            target_reposed_vertices_tiled = np.tile(target_reposed_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_reposed_vertices_samples_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices_samples,
                target_reposed_vertices_tiled)
            pvet_sc_per_sample = np.linalg.norm(pred_reposed_vertices_samples_sc - target_reposed_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pvet_sc_sample = np.argmin(np.mean(pvet_sc_per_sample, axis=-1))
            pvet_sc_samples_min_batch = pvet_sc_per_sample[min_pvet_sc_sample]  # (6890,)
            metric_sums['pve-ts_sc_samples_min'] += np.sum(pvet_sc_samples_min_batch)
            per_frame_metrics['pve-ts_sc_samples_min'].append(np.mean(pvet_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        if 'mpjpes' in metrics_to_track:
            mpjpe_batch = np.linalg.norm(pred_joints_h36mlsp_mode - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            metric_sums['mpjpes'] += np.sum(mpjpe_batch)  # scalar
            per_frame_metrics['mpjpes'].append(np.mean(mpjpe_batch, axis=-1))

        if 'mpjpes_samples_min' in metrics_to_track:
            mpjpe_per_sample = np.linalg.norm(pred_joints_h36mlsp_samples - target_joints_h36mlsp, axis=-1)  # (num samples, 14)
            min_mpjpe_sample = np.argmin(np.mean(mpjpe_per_sample, axis=-1))
            mpjpe_samples_min_batch = mpjpe_per_sample[min_mpjpe_sample]  # (14,)
            metric_sums['mpjpes_samples_min'] += np.sum(mpjpe_samples_min_batch)
            per_frame_metrics['mpjpes_samples_min'].append(np.mean(mpjpe_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'mpjpes_sc' in metrics_to_track:
            pred_joints_h36mlsp_sc = scale_and_translation_transform_batch(
                pred_joints_h36mlsp_mode,
                target_joints_h36mlsp)
            mpjpe_sc_batch = np.linalg.norm(
                pred_joints_h36mlsp_sc - target_joints_h36mlsp,
                axis=-1)  # (bs, 14)
            metric_sums['mpjpes_sc'] += np.sum(mpjpe_sc_batch)  # scalar
            per_frame_metrics['mpjpes_sc'].append(np.mean(mpjpe_sc_batch, axis=-1))

        if 'mpjpes_sc_samples_min' in metrics_to_track:
            target_joints_h36mlsp_tiled = np.tile(target_joints_h36mlsp, (num_pred_samples, 1, 1))  # (num samples, 14, 3)
            pred_joints_h36mlsp_samples_sc = scale_and_translation_transform_batch(
                pred_joints_h36mlsp_samples,
                target_joints_h36mlsp_tiled)
            mpjpe_sc_per_sample = np.linalg.norm(pred_joints_h36mlsp_samples_sc - target_joints_h36mlsp_tiled, axis=-1)  # (num samples, 14)
            min_mpjpe_sc_sample = np.argmin(np.mean(mpjpe_sc_per_sample, axis=-1))
            mpjpe_sc_samples_min_batch = mpjpe_sc_per_sample[min_mpjpe_sc_sample]  # (14,)
            metric_sums['mpjpes_sc_samples_min'] += np.sum(mpjpe_sc_samples_min_batch)
            per_frame_metrics['mpjpes_sc_samples_min'].append(np.mean(mpjpe_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Procrustes analysis
        if 'mpjpes_pa' in metrics_to_track:
            pred_joints_h36mlsp_pa = compute_similarity_transform_batch_numpy(pred_joints_h36mlsp_mode, target_joints_h36mlsp)
            mpjpe_pa_batch = np.linalg.norm(pred_joints_h36mlsp_pa - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            metric_sums['mpjpes_pa'] += np.sum(mpjpe_pa_batch)  # scalar
            per_frame_metrics['mpjpes_pa'].append(np.mean(mpjpe_pa_batch, axis=-1))

        if 'mpjpes_pa_samples_min' in metrics_to_track:
            target_joints_h36mlsp_tiled = np.tile(target_joints_h36mlsp, (num_pred_samples, 1, 1))  # (num samples, 14, 3)
            pred_joints_h36mlsp_samples_pa = compute_similarity_transform_batch_numpy(
                pred_joints_h36mlsp_samples,
                target_joints_h36mlsp_tiled)
            mpjpe_pa_per_sample = np.linalg.norm(pred_joints_h36mlsp_samples_pa - target_joints_h36mlsp_tiled, axis=-1)  # (num samples, 14)
            min_mpjpe_pa_sample = np.argmin(np.mean(mpjpe_pa_per_sample, axis=-1))
            mpjpe_pa_samples_min_batch = mpjpe_pa_per_sample[min_mpjpe_pa_sample]  # (14,)
            metric_sums['mpjpes_pa_samples_min'] += np.sum(mpjpe_pa_samples_min_batch)
            per_frame_metrics['mpjpes_pa_samples_min'].append(np.mean(mpjpe_pa_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # ---------------- 3D Sample Distance from Mean (i.e. Variance) Metrics -----------
        if 'verts_samples_dist_from_mean' in metrics_to_track:
            verts_samples_mean = pred_vertices_samples.mean(axis=0)  # (6890, 3)
            verts_samples_dist_from_mean = np.linalg.norm(pred_vertices_samples - verts_samples_mean, axis=-1)  # (num samples, 6890)
            metric_sums['verts_samples_dist_from_mean'] += verts_samples_dist_from_mean.sum()
            per_frame_metrics['verts_samples_dist_from_mean'].append(verts_samples_dist_from_mean.mean()[None])  # (1,)

        if 'joints3D_coco_samples_dist_from_mean' in metrics_to_track:
            joints3D_coco_samples_mean = pred_joints_coco_samples.mean(axis=0)  # (17, 3)
            joints3D_coco_samples_dist_from_mean = np.linalg.norm(pred_joints_coco_samples - joints3D_coco_samples_mean, axis=-1)  # (num samples, 17)
            metric_sums['joints3D_coco_samples_dist_from_mean'] += joints3D_coco_samples_dist_from_mean.sum()  # scalar
            per_frame_metrics['joints3D_coco_samples_dist_from_mean'].append(joints3D_coco_samples_dist_from_mean.mean()[None])  # (1,)

        if 'joints3D_coco_invis_samples_dist_from_mean' in metrics_to_track:
            # (In)visibility of specific joints determined by HRNet 2D joint predictions and confidence scores.
            hrnet_joints2D_coco_invis = np.logical_not(hrnet_joints2D_coco_vis[0])  # (17,)

            if np.any(hrnet_joints2D_coco_invis):
                joints3D_coco_invis_samples = pred_joints_coco_samples[:, hrnet_joints2D_coco_invis, :]  # (num samples, num invis joints, 3)
                joints3D_coco_invis_samples_mean = joints3D_coco_invis_samples.mean(axis=0)  # (num_invis_joints, 3)
                joints3D_coco_invis_samples_dist_from_mean = np.linalg.norm(joints3D_coco_invis_samples - joints3D_coco_invis_samples_mean,
                                                                            axis=-1)  # (num samples, num_invis_joints)

                metric_sums['joints3D_coco_invis_samples_dist_from_mean'] += joints3D_coco_invis_samples_dist_from_mean.sum()  # scalar
                metric_sums['num_invis_joints3Dsamples'] += np.prod(joints3D_coco_invis_samples_dist_from_mean.shape)
                per_frame_metrics['joints3D_coco_invis_samples_dist_from_mean'].append(joints3D_coco_invis_samples_dist_from_mean.mean()[None])  # (1,)
            else:
                per_frame_metrics['joints3D_coco_invis_samples_dist_from_mean'].append(np.zeros(1))

        # -------------------------------- 2D Metrics ---------------------------
        # Using JRNet 2D joints as target, rather than GT
        if 'hrnet_joints2D_l2es' in metrics_to_track:
            hrnet_joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco_mode[:, hrnet_joints2D_coco_vis[0], :] - hrnet_joints2D_coco[:, hrnet_joints2D_coco_vis[0], :],
                                                      axis=-1)  # (1, num vis joints)
            assert hrnet_joints2D_l2e_batch.shape[1] == hrnet_joints2D_coco_vis.sum()

            metric_sums['hrnet_joints2D_l2es'] += np.sum(hrnet_joints2D_l2e_batch)  # scalar
            metric_sums['num_vis_hrnet_joints2D'] += hrnet_joints2D_l2e_batch.shape[1]
            per_frame_metrics['hrnet_joints2D_l2es'].append(np.mean(hrnet_joints2D_l2e_batch, axis=-1))  # (1,)

        # -------------------------------- 2D Metrics after Averaging over Samples ---------------------------
        if 'hrnet_joints2Dsamples_l2es' in metrics_to_track:
            hrnet_joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples[:, hrnet_joints2D_coco_vis[0], :] - hrnet_joints2D_coco[:, hrnet_joints2D_coco_vis[0], :],
                                                             axis=-1)  # (num_samples, num vis joints)
            assert hrnet_joints2Dsamples_l2e_batch.shape[1] == hrnet_joints2D_coco_vis.sum()

            metric_sums['hrnet_joints2Dsamples_l2es'] += np.sum(hrnet_joints2Dsamples_l2e_batch)  # scalar
            metric_sums['num_vis_hrnet_joints2Dsamples'] += np.prod(hrnet_joints2Dsamples_l2e_batch.shape)
            per_frame_metrics['hrnet_joints2Dsamples_l2es'].append(np.mean(hrnet_joints2Dsamples_l2e_batch, axis=-1, keepdims=True))  # (1,)

        metric_sums['num_datapoints'] += target_pose.shape[0]

        fname_per_frame.append(fname)
        pose_per_frame.append(np.concatenate([pred_glob_rotmat_mode.cpu().detach().numpy(),
                                              pred_pose_rotmats_mode.cpu().detach().numpy()],
                                             axis=1))
        shape_per_frame.append(pred_shape_mode.cpu().detach().numpy())
        cam_per_frame.append(pred_cam_wp.cpu().detach().numpy())

        # ------------------------------- VISUALISE -------------------------------
        if vis_every_n_batches is not None and batch_num % vis_every_n_batches == 0:
            vis_img = samples_batch['vis_img'].numpy()
            vis_img = np.transpose(vis_img, [0, 2, 3, 1])

            pred_cam_t = out['pred_cam_t'][0, 0, :].cpu().detach().numpy()

            # Uncertainty Computation
            # Uncertainty computed by sampling + average distance from mean
            avg_vertices_distance_from_mean, avg_vertices_sc_distance_from_mean = compute_vertex_uncertainties_from_samples(
                vertices_samples=pred_vertices_samples,
                target_vertices=target_vertices)
            if save_per_frame_uncertainty:
                vertices_uncertainty_per_frame.append(avg_vertices_distance_from_mean)

            # Render predicted meshes
            body_vis_rgb_mode = renderer(vertices=pred_vertices_mode[0],
                                         camera_translation=pred_cam_t,
                                         image=vis_img[0],
                                         unnormalise_img=False)
            body_vis_rgb_mode_rot = renderer(vertices=pred_vertices_mode[0],
                                             camera_translation=pred_cam_t,
                                             image=vis_img[0],
                                             unnormalise_img=False,
                                             angle=np.pi/2.,
                                             axis=[0., 1., 0.])

            reposed_body_vis_rgb_mean = renderer(vertices=pred_reposed_vertices_mean[0],
                                                 camera_translation=pred_cam_t,
                                                 image=vis_img[0],
                                                 unnormalise_img=False)
            reposed_body_vis_rgb_mean_rot = renderer(vertices=pred_reposed_vertices_mean[0],
                                                     camera_translation=pred_cam_t,
                                                     image=vis_img[0],
                                                     unnormalise_img=False,
                                                     angle=np.pi / 2.,
                                                     axis=[0., 1., 0.])

            body_vis_rgb_samples = []
            body_vis_rgb_rot_samples = []
            for i in range(num_pred_samples):
                body_vis_rgb_samples.append(renderer(vertices=pred_vertices_samples[i],
                                                     camera_translation=pred_cam_t,
                                                     image=vis_img[0],
                                                     unnormalise_img=False))
                body_vis_rgb_rot_samples.append(renderer(vertices=pred_vertices_samples[i],
                                                         camera_translation=pred_cam_t,
                                                         image=vis_img[0],
                                                         unnormalise_img=False,
                                                         angle=np.pi / 2.,
                                                         axis=[0., 1., 0.]))

            # Save samples
            samples_save_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_samples.npy')
            np.save(samples_save_path, pred_vertices_samples)

            # ------------------ Model Prediction, Error and Uncertainty Figure ------------------
            num_row = 5
            num_col = 6
            subplot_count = 1
            plt.figure(figsize=(20, 20))

            # Plot image and mask vis
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(vis_img[0])
            subplot_count += 1

            # Plot pred vertices 2D and body render overlaid over input
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(vis_img[0])
            plt.scatter(pred_vertices2D_mode[0, :, 0],
                        pred_vertices2D_mode[0, :, 1],
                        c='r', s=0.01)
            subplot_count += 1

            # Plot body render overlaid on vis image
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_mode)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_mode_rot)
            subplot_count += 1

            # Plot reposed body render
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(reposed_body_vis_rgb_mean)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(reposed_body_vis_rgb_mean_rot)
            subplot_count += 1

            if 'pves_sc' in metrics_to_track:
                # Plot PVE-SC pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_vertices_sc[0, :, 0],
                            pred_vertices_sc[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 0],
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 2], # Equivalent to Rotated 90° about y axis
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][0]))
                subplot_count += 1

            if 'pves_pa' in metrics_to_track:
                # Plot PVE-PA pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-PA')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_vertices_pa[0, :, 0],
                            pred_vertices_pa[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 0],
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 2], # Equivalent to Rotated 90° about y axis
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][0]))
                subplot_count += 1
            
            if 'pve-ts_sc' in metrics_to_track:
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-T-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_reposed_vertices[0, :, 0],
                            target_reposed_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_reposed_vertices_sc[0, :, 0],
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.03, clip=True)
                plt.scatter(pred_reposed_vertices_sc[0, :, 0],
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pvet_scale_corrected_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.03, clip=True)
                plt.scatter(pred_reposed_vertices_sc[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pvet_scale_corrected_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][0]))
                subplot_count += 1

            # Plot per-vertex uncertainties
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.text(0.5, 0.5, s='Uncertainty for\nPVE')
            subplot_count += 1
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            plt.scatter(pred_vertices_sc[0, :, 0],
                        pred_vertices_sc[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean,
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            plt.scatter(pred_vertices_sc[0, :, 2],
                        pred_vertices_sc[0, :, 1],
                        s=0.05,
                        c=avg_vertices_sc_distance_from_mean,
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.text(0.5, 0.5, s='Uncertainty for\nPVE-SC')
            subplot_count += 1
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            plt.scatter(pred_vertices_pa[0, :, 0],
                        pred_vertices_pa[0, :, 1],
                        s=0.05,
                        c=avg_vertices_sc_distance_from_mean,
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            plt.scatter(pred_vertices_pa[0, :, 2],
                        # Equivalent to Rotated 90° about y axis
                        pred_vertices_pa[0, :, 1],
                        s=0.05,
                        c=avg_vertices_sc_distance_from_mean,
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            save_fig_path = os.path.join(save_path, fname[0])
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()

            # ------------------ Samples from Predicted Distribution Figure ------------------
            num_subplots = num_samples_to_visualise * 2
            num_row = 5
            num_col = math.ceil(num_subplots / float(num_row))

            subplot_count = 1
            plt.figure(figsize=(20, 20))

            # Plot mode prediction
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_mode)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_mode_rot)
            subplot_count += 1

            # Plot samples from predicted distribution
            for i in range(2 * num_samples_to_visualise):
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(body_vis_rgb_samples[i])
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(body_vis_rgb_rot_samples[i])
                subplot_count += 1

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            save_fig_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_samples.png')
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()




    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    print('\n--- Check Pred save Shapes ---')
    fname_per_frame = np.concatenate(fname_per_frame, axis=0)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)
    print(fname_per_frame.shape)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)
    print(pose_per_frame.shape)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)
    print(shape_per_frame.shape)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)
    print(cam_per_frame.shape)

    if vis_every_n_batches is not None and save_per_frame_uncertainty:
        vertices_uncertainty_per_frame = np.stack(vertices_uncertainty_per_frame, axis=0)
        np.save(os.path.join(save_path, 'vertices_uncertainty_per_frame.npy'), vertices_uncertainty_per_frame)
        print(vertices_uncertainty_per_frame.shape)

    final_metrics = {}
    for metric_type in metrics_to_track:

        if metric_type == 'hrnet_joints2D_l2es':
            joints2D_l2e = metric_sums['hrnet_joints2D_l2es'] / metric_sums['num_vis_hrnet_joints2D']
            final_metrics[metric_type] = joints2D_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_hrnet_joints2D'])
        elif metric_type == 'hrnet_joints2D_l2es_best_j2d_sample':
            joints2D_l2e_best_j2d_sample = metric_sums['hrnet_joints2D_l2es_best_j2d_sample'] / metric_sums['num_vis_hrnet_joints2D']
            final_metrics[metric_type] = joints2D_l2e_best_j2d_sample
        elif metric_type == 'hrnet_joints2Dsamples_l2es':
            joints2Dsamples_l2e = metric_sums['hrnet_joints2Dsamples_l2es'] / metric_sums['num_vis_hrnet_joints2Dsamples']
            final_metrics[metric_type] = joints2Dsamples_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_hrnet_joints2Dsamples'])

        elif metric_type == 'verts_samples_dist_from_mean':
            final_metrics[metric_type] = metric_sums[metric_type] / (metric_sums['num_datapoints'] * num_pred_samples * 6890)
        elif metric_type == 'joints3D_coco_samples_dist_from_mean':
            final_metrics[metric_type] = metric_sums[metric_type] / (metric_sums['num_datapoints'] * num_pred_samples * 17)
        elif metric_type == 'joints3D_coco_invis_samples_dist_from_mean':
            if metric_sums['num_invis_joints3Dsamples'] > 0:
                final_metrics[metric_type] = metric_sums[metric_type] / metric_sums['num_invis_joints3Dsamples']
            else:
                print('No invisible 3D COCO joints!')

        else:
            if 'pves' in metric_type:
                num_per_sample = 6890
            elif 'mpjpes' in metric_type:
                num_per_sample = 14
            # print('Check total samples:', metric_type, num_per_sample, self.total_samples)
            final_metrics[metric_type] = metric_sums[metric_type] / (metric_sums['num_datapoints'] * num_per_sample)

    print('\n---- Metrics ----')
    for metric in final_metrics.keys():
        if final_metrics[metric] > 0.3:
            mult = 1
        else:
            mult = 1000
        print(metric, '{:.2f}'.format(final_metrics[metric] * mult))  # Converting from metres to millimetres

    print('\n---- Check metric save shapes ----')
    for metric_type in metrics_to_track:
        per_frame = np.concatenate(per_frame_metrics[metric_type], axis=0)
        print(metric_type, per_frame.shape)
        np.save(os.path.join(save_path, metric_type + '_per_frame.npy'), per_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
    parser.add_argument('--gpu', default='0', type=str, help='GPU')
    parser.add_argument('--num_samples', '-N', type=int, default=25, help='Number of test samples to evaluate with')
    parser.add_argument('--use_subset', '-S',action='store_true')

    args = parser.parse_args()

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)
    model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
    model.eval()
    model_cfg.defrost()
    model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples + 1
    model_cfg.freeze()

    # Renderer for visualisation
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Setup evaluation dataset
    if args.use_subset:
        selected_fnames = subsets.PW3D_OCCLUDED_JOINTS
        vis_every_n_batches = 1
        vis_joints_threshold = 0.8
    else:
        selected_fnames = None
        vis_every_n_batches = 1000
        vis_joints_threshold = 0.6

    dataset_path = '/scratches/nazgul_2/as2562/datasets/3DPW/test'
    dataset = PW3DEvalDataset(dataset_path,
                              img_wh=model_cfg.MODEL.IMAGE_SIZE,
                              selected_fnames=selected_fnames,
                              visible_joints_threshold=vis_joints_threshold)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc', 'mpjpes', 'mpjpes_sc', 'mpjpes_pa']
    metrics.extend([metric + '_samples_min' for metric in metrics ])
    metrics.extend(['verts_samples_dist_from_mean', 'joints3D_coco_samples_dist_from_mean', 'joints3D_coco_invis_samples_dist_from_mean'])
    metrics.append('hrnet_joints2D_l2es')
    metrics.append('hrnet_joints2Dsamples_l2es')

    save_path = '/scratch/as2562/ProHMR/evaluations/3dpw_{}_samples'.format(args.num_samples)
    if args.use_subset:
        save_path += '_selected_fnames_occluded_joints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Saving to:", save_path)

    # Run evaluation
    evaluate_3dpw(model=model,
                  eval_dataset=dataset,
                  metrics_to_track=metrics,
                  device=device,
                  save_path=save_path,
                  num_pred_samples=args.num_samples,
                  num_workers=4,
                  pin_memory=True,
                  vis_every_n_batches=vis_every_n_batches,
                  renderer=renderer)
