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

import my_config

from prohmr.configs import get_config, prohmr_config
from prohmr.models import ProHMR
from prohmr.models.smpl_mine import SMPL
from prohmr.utils.pose_utils import compute_similarity_transform_batch, scale_and_translation_transform_batch
from prohmr.utils.geometry import undo_keypoint_normalisation, orthographic_project_torch
from prohmr.datasets.pw3d_eval_dataset import PW3DEvalDataset

import subsets


def evaluate_3dpw(model,
                  eval_dataset,
                  metrics,
                  device,
                  vis_save_path,
                  num_pred_samples,
                  num_workers=4,
                  pin_memory=True,
                  vis_every_n_batches=1000):

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
    for metric in metrics:
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

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        if batch_num == 2:
            break
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input'].to(device)
        target_pose = samples_batch['pose'].to(device)
        target_shape = samples_batch['shape'].to(device)
        target_gender = samples_batch['gender'][0]
        hrnet_joints2D_coco = samples_batch['hrnet_kps']
        hrnet_joints2D_coco_vis = samples_batch['hrnet_kps_vis']
        print('HRNET KPS SHAPES', hrnet_joints2D_coco.shape, hrnet_joints2D_coco_vis.shape)
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
        out = model(input)
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
        if 'pve' in metrics:
            pve_batch = np.linalg.norm(pred_vertices_mode - target_vertices,
                                       axis=-1)  # (bs, 6890)
            metric_sums['pve'] += np.sum(pve_batch)  # scalar
            per_frame_metrics['pve'].append(np.mean(pve_batch, axis=-1))

        if 'pve_samples_min' in metrics:
            pve_per_sample = np.linalg.norm(pred_vertices_samples - target_vertices, axis=-1)  # (num samples, 6890)
            min_pve_sample = np.argmin(np.mean(pve_per_sample, axis=-1))
            pve_samples_min_batch = pve_per_sample[min_pve_sample]  # (6890,)
            metric_sums['pve_samples_min'] += np.sum(pve_samples_min_batch)
            per_frame_metrics['pve_samples_min'].append(np.mean(pve_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'pve_sc' in metrics:
            pred_vertices_scale_corrected = scale_and_translation_transform_batch(
                pred_vertices_mode,
                target_vertices)
            pve_sc_batch = np.linalg.norm(
                pred_vertices_scale_corrected - target_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pve_sc'] += np.sum(pve_sc_batch)  # scalar
            per_frame_metrics['pve_sc'].append(np.mean(pve_sc_batch, axis=-1))

        if 'pve_sc_samples_min' in metrics:
            target_vertices_tiled = np.tile(target_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_scale_corrected = scale_and_translation_transform_batch(
                pred_vertices_samples,
                target_vertices_tiled)
            pve_sc_per_sample = np.linalg.norm(pred_vertices_samples_scale_corrected - target_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pve_sc_sample = np.argmin(np.mean(pve_sc_per_sample, axis=-1))
            pve_sc_samples_min_batch = pve_sc_per_sample[min_pve_sc_sample]  # (6890,)
            metric_sums['pve_sc_samples_min'] += np.sum(pve_sc_samples_min_batch)
            per_frame_metrics['pve_sc_samples_min'].append(np.mean(pve_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Procrustes analysis
        if 'pve_pa' in metrics:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices_mode, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bs, 6890)
            metric_sums['pve_pa'] += np.sum(pve_pa_batch)  # scalar
            per_frame_metrics['pve_pa'].append(np.mean(pve_pa_batch, axis=-1))

        if 'pve_pa_samples_min' in metrics:
            target_vertices_tiled = np.tile(target_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_pa = compute_similarity_transform_batch(
                pred_vertices_samples,
                target_vertices_tiled)
            pve_pa_per_sample = np.linalg.norm(pred_vertices_samples_pa - target_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pve_pa_sample = np.argmin(np.mean(pve_pa_per_sample, axis=-1))
            pve_pa_samples_min_batch = pve_pa_per_sample[min_pve_pa_sample]  # (6890,)
            metric_sums['pve_pa_samples_min'] += np.sum(pve_pa_samples_min_batch)
            per_frame_metrics['pve_pa_samples_min'].append(np.mean(pve_pa_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        if 'pve-t' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices_mean - target_reposed_vertices, axis=-1)
            metric_sums['pve-t'] += np.sum(pvet_batch)  # scalar
            per_frame_metrics['pve-t'].append(np.mean(pvet_batch, axis=-1))

        if 'pve-t_samples_min' in metrics:
            pvet_per_sample = np.linalg.norm(pred_reposed_vertices_samples - target_reposed_vertices, axis=-1)  # (num samples, 6890)
            min_pvet_sample = np.argmin(np.mean(pvet_per_sample, axis=-1))
            pvet_samples_min_batch = pvet_per_sample[min_pvet_sample]  # (6890,)
            metric_sums['pve-t_samples_min'] += np.sum(pvet_samples_min_batch)
            per_frame_metrics['pve-t_samples_min'].append(np.mean(pvet_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'pve-t_sc' in metrics:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices_mean,
                target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(
                pred_reposed_vertices_sc - target_reposed_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pve-t_sc'] += np.sum(pvet_scale_corrected_batch)  # scalar
            per_frame_metrics['pve-t_sc'].append(np.mean(pvet_scale_corrected_batch, axis=-1))

        if 'pve-t_sc_samples_min' in metrics:
            target_reposed_vertices_tiled = np.tile(target_reposed_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_reposed_vertices_samples_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices_samples,
                target_reposed_vertices_tiled)
            pvet_sc_per_sample = np.linalg.norm(pred_reposed_vertices_samples_sc - target_reposed_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pvet_sc_sample = np.argmin(np.mean(pvet_sc_per_sample, axis=-1))
            pvet_sc_samples_min_batch = pvet_sc_per_sample[min_pvet_sc_sample]  # (6890,)
            metric_sums['pve-t_sc_samples_min'] += np.sum(pvet_sc_samples_min_batch)
            per_frame_metrics['pve-t_sc_samples_min'].append(np.mean(pvet_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        if 'mpjpe' in metrics:
            mpjpe_batch = np.linalg.norm(pred_joints_h36mlsp_mode - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            metric_sums['mpjpe'] += np.sum(mpjpe_batch)  # scalar
            per_frame_metrics['mpjpe'].append(np.mean(mpjpe_batch, axis=-1))

        if 'mpjpe_samples_min' in metrics:
            mpjpe_per_sample = np.linalg.norm(pred_joints_h36mlsp_samples - target_joints_h36mlsp, axis=-1)  # (num samples, 14)
            min_mpjpe_sample = np.argmin(np.mean(mpjpe_per_sample, axis=-1))
            mpjpe_samples_min_batch = mpjpe_per_sample[min_mpjpe_sample]  # (14,)
            metric_sums['mpjpe_samples_min'] += np.sum(mpjpe_samples_min_batch)
            per_frame_metrics['mpjpe_samples_min'].append(np.mean(mpjpe_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'mpjpe_sc' in metrics:
            pred_joints_h36mlsp_sc = scale_and_translation_transform_batch(
                pred_joints_h36mlsp_mode,
                target_joints_h36mlsp)
            mpjpe_sc_batch = np.linalg.norm(
                pred_joints_h36mlsp_sc - target_joints_h36mlsp,
                axis=-1)  # (bs, 14)
            metric_sums['mpjpe_sc'] += np.sum(mpjpe_sc_batch)  # scalar
            per_frame_metrics['mpjpe_sc'].append(np.mean(mpjpe_sc_batch, axis=-1))

        if 'mpjpe_sc_samples_min' in metrics:
            target_joints_h36mlsp_tiled = np.tile(target_joints_h36mlsp, (num_pred_samples, 1, 1))  # (num samples, 14, 3)
            pred_joints_h36mlsp_samples_sc = scale_and_translation_transform_batch(
                pred_joints_h36mlsp_samples,
                target_joints_h36mlsp_tiled)
            mpjpe_sc_per_sample = np.linalg.norm(pred_joints_h36mlsp_samples_sc - target_joints_h36mlsp_tiled, axis=-1)  # (num samples, 14)
            min_mpjpe_sc_sample = np.argmin(np.mean(mpjpe_sc_per_sample, axis=-1))
            mpjpe_sc_samples_min_batch = mpjpe_sc_per_sample[min_mpjpe_sc_sample]  # (14,)
            metric_sums['mpjpe_sc_samples_min'] += np.sum(mpjpe_sc_samples_min_batch)
            per_frame_metrics['mpjpe_sc_samples_min'].append(np.mean(mpjpe_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Procrustes analysis
        if 'mpjpe_pa' in metrics:
            pred_joints_h36mlsp_pa = compute_similarity_transform_batch(pred_joints_h36mlsp_mode, target_joints_h36mlsp)
            mpjpe_pa_batch = np.linalg.norm(pred_joints_h36mlsp_pa - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            metric_sums['mpjpe_pa'] += np.sum(mpjpe_pa_batch)  # scalar
            per_frame_metrics['mpjpe_pa'].append(np.mean(mpjpe_pa_batch, axis=-1))

        if 'mpjpe_pa_samples_min' in metrics:
            target_joints_h36mlsp_tiled = np.tile(target_joints_h36mlsp, (num_pred_samples, 1, 1))  # (num samples, 14, 3)
            pred_joints_h36mlsp_samples_pa = compute_similarity_transform_batch(
                pred_joints_h36mlsp_samples,
                target_joints_h36mlsp_tiled)
            mpjpe_pa_per_sample = np.linalg.norm(pred_joints_h36mlsp_samples_pa - target_joints_h36mlsp_tiled, axis=-1)  # (num samples, 14)
            min_mpjpe_pa_sample = np.argmin(np.mean(mpjpe_pa_per_sample, axis=-1))
            mpjpe_pa_samples_min_batch = mpjpe_pa_per_sample[min_mpjpe_pa_sample]  # (14,)
            metric_sums['mpjpe_pa_samples_min'] += np.sum(mpjpe_pa_samples_min_batch)
            per_frame_metrics['mpjpe_pa_samples_min'].append(np.mean(mpjpe_pa_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # ---------------- 3D Sample Distance from Mean (i.e. Variance) Metrics -----------
        if 'verts_samples_dist_from_mean' in metrics:
            verts_samples_mean = pred_vertices_samples.mean(axis=0)  # (6890, 3)
            verts_samples_dist_from_mean = np.linalg.norm(pred_vertices_samples - verts_samples_mean, axis=-1)  # (num samples, 6890)
            metric_sums['verts_samples_dist_from_mean'] += verts_samples_dist_from_mean.sum()
            per_frame_metrics['verts_samples_dist_from_mean'].append(verts_samples_dist_from_mean.mean()[None])  # (1,)

        if 'joints3D_coco_samples_dist_from_mean' in metrics:
            joints3D_coco_samples_mean = pred_joints_coco_samples.mean(axis=0)  # (17, 3)
            joints3D_coco_samples_dist_from_mean = np.linalg.norm(pred_joints_coco_samples - joints3D_coco_samples_mean, axis=-1)  # (num samples, 17)
            metric_sums['joints3D_coco_samples_dist_from_mean'] += joints3D_coco_samples_dist_from_mean.sum()  # scalar
            per_frame_metrics['joints3D_coco_samples_dist_from_mean'].append(joints3D_coco_samples_dist_from_mean.mean()[None])  # (1,)

        if 'joints3D_coco_invis_samples_dist_from_mean' in metrics:
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
        if 'hrnet_joints2D_l2es' in metrics:
            hrnet_joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco_mode[:, hrnet_joints2D_coco_vis[0], :] - hrnet_joints2D_coco[:, hrnet_joints2D_coco_vis[0], :],
                                                      axis=-1)  # (1, num vis joints)
            print('hrnet_joints2D_l2e_batch', hrnet_joints2D_l2e_batch.shape)
            assert hrnet_joints2D_l2e_batch.shape[1] == hrnet_joints2D_coco_vis.sum()

            metric_sums['hrnet_joints2D_l2es'] += np.sum(hrnet_joints2D_l2e_batch)  # scalar
            metric_sums['num_vis_hrnet_joints2D'] += hrnet_joints2D_l2e_batch.shape[1]
            per_frame_metrics['hrnet_joints2D_l2es'].append(np.mean(hrnet_joints2D_l2e_batch, axis=-1))  # (1,)

        # -------------------------------- 2D Metrics after Averaging over Samples ---------------------------
        if 'hrnet_joints2Dsamples_l2es' in metrics:
            hrnet_joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples[:, hrnet_joints2D_coco_vis[0], :] - hrnet_joints2D_coco[:, hrnet_joints2D_coco_vis[0], :],
                                                             axis=-1)  # (num_samples, num vis joints)
            print('hrnet_joints2Dsamples_l2e_batch', hrnet_joints2Dsamples_l2e_batch.shape)
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
        # if vis_every_n_batches is not None:
        #     if batch_num % vis_every_n_batches == 0:
        #         vis_imgs = samples_batch['vis_img'].numpy()
        #         vis_imgs = np.transpose(vis_imgs, [0, 2, 3, 1])
        #
        #         plt.figure(figsize=(16, 12))
        #         plt.subplot(341)
        #         plt.gca().axis('off')
        #         plt.imshow(vis_imgs[0])
        #
        #         plt.subplot(342)
        #         plt.gca().axis('off')
        #         plt.imshow(vis_imgs[0])
        #         plt.scatter(pred_vertices_projected2d[0, :, 0], pred_vertices_projected2d[0, :, 1], s=0.1, c='r')
        #
        #         plt.subplot(343)
        #         plt.gca().axis('off')
        #         plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.2, c='b')
        #         # plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.1, c='r')
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #
        #         plt.subplot(344)
        #         plt.gca().axis('off')
        #         plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.2, c='b')
        #         plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.1, c='r')
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #         plt.text(-0.6, -0.8, s='PVE: {:.4f}'.format(pve_per_frame[-1][0]))
        #
        #         plt.subplot(345)
        #         plt.gca().axis('off')
        #         plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.2,
        #                     c='b')
        #         plt.scatter(pred_vertices_scale_corrected[0, :, 0],
        #                     pred_vertices_scale_corrected[0, :, 1], s=0.1,
        #                     c='r')
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #         plt.text(-0.6, -0.8, s='PVE-SC: {:.4f}'.format(pve_sc_per_frame[-1][0]))
        #
        #         plt.subplot(346)
        #         plt.gca().axis('off')
        #         plt.scatter(target_vertices[0, :, 2], target_vertices[0, :, 1], s=0.2,
        #                     c='b')
        #         plt.scatter(pred_vertices_scale_corrected[0, :, 2],
        #                     pred_vertices_scale_corrected[0, :, 1], s=0.1,
        #                     c='r')
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #
        #         plt.subplot(347)
        #         plt.gca().axis('off')
        #         plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
        #         plt.scatter(pred_vertices_pa[0, :, 0], pred_vertices_pa[0, :, 1], s=0.1, c='r')
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #         plt.text(-0.6, -0.8, s='PVE-PA: {:.4f}'.format(pve_pa_per_frame[-1][0]))
        #
        #         plt.subplot(348)
        #         plt.gca().axis('off')
        #         plt.scatter(target_vertices[0, :, 2], target_vertices[0, :, 1], s=0.2, c='b')
        #         plt.scatter(pred_vertices_pa[0, :, 2], pred_vertices_pa[0, :, 1], s=0.1, c='r')
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #
        #         plt.subplot(349)
        #         plt.gca().axis('off')
        #         plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
        #         plt.scatter(pred_reposed_vertices_sc[0, :, 0], pred_reposed_vertices_sc[0, :, 1], s=0.1, c='r')
        #         plt.gca().set_aspect('equal', adjustable='box')
        #
        #         plt.subplot(3, 4, 10)
        #         plt.gca().axis('off')
        #         for j in range(num_joints3d_h36mlsp):
        #             plt.scatter(pred_joints_h36mlsp[0, j, 0], pred_joints_h36mlsp[0, j, 1], c='r')
        #             plt.scatter(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], c='b')
        #             plt.text(pred_joints_h36mlsp[0, j, 0], pred_joints_h36mlsp[0, j, 1], s=str(j))
        #             plt.text(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], s=str(j))
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #
        #         plt.subplot(3, 4, 11)
        #         plt.gca().axis('off')
        #         for j in range(num_joints3d_h36mlsp):
        #             plt.scatter(pred_joints_h36mlsp_sc[0, j, 0],
        #                         pred_joints_h36mlsp_sc[0, j, 1], c='r')
        #             plt.scatter(target_joints_h36mlsp[0, j, 0],
        #                         target_joints_h36mlsp[0, j, 1], c='b')
        #             plt.text(pred_joints_h36mlsp_sc[0, j, 0],
        #                      pred_joints_h36mlsp_sc[0, j, 1], s=str(j))
        #             plt.text(target_joints_h36mlsp[0, j, 0],
        #                      target_joints_h36mlsp[0, j, 1], s=str(j))
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #
        #         plt.subplot(3, 4, 12)
        #         plt.gca().axis('off')
        #         for j in range(num_joints3d_h36mlsp):
        #             plt.scatter(pred_joints_h36mlsp_pa[0, j, 0], pred_joints_h36mlsp_pa[0, j, 1], c='r')
        #             plt.scatter(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], c='b')
        #             plt.text(pred_joints_h36mlsp_pa[0, j, 0], pred_joints_h36mlsp_pa[0, j, 1], s=str(j))
        #             plt.text(target_joints_h36mlsp[0, j, 0], target_joints_h36mlsp[0, j, 1], s=str(j))
        #         plt.gca().invert_yaxis()
        #         plt.gca().set_aspect('equal', adjustable='box')
        #
        #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #         plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #
        #         # plt.show()
        #         save_fig_path = os.path.join(vis_save_path, fnames[0])
        #         plt.savefig(save_fig_path, bbox_inches='tight')
        #         plt.close()

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
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

    final_metrics = {}
    for metric_type in metrics:

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
            if 'pve' in metric_type:
                num_per_sample = 6890
            elif 'mpjpe' in metric_type:
                num_per_sample = 14
            # print('Check total samples:', metric_type, num_per_sample, self.total_samples)
            final_metrics[metric_type] = metric_sums[metric_type] / (num_datapoints * num_per_sample)

    print('\n---- METRICS ----')
    for metric in final_metrics.keys():
        if final_metrics[metric] > 0.3:
            mult = 1
        else:
            mult = 1000
        print(metric, '{:.2f}'.format(final_metrics[metric] * mult))  # Converting from metres to millimetres


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
    parser.add_argument('--gpu', default='0', type=str, help='GPU')
    parser.add_argument('--num_samples', type=int, default=25, help='Number of test samples to evaluate with')
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

    # Setup evaluation dataset
    # selected_fnames = subsets.PW3D_OCCLUDED_JOINTS
    selected_fnames = None
    print('Selected fnames:', selected_fnames)
    if selected_fnames is not None:
        vis_every_n_batches = 1
        vis_joints_threshold = 0.8
    else:
        vis_every_n_batches = 1000
        vis_joints_threshold = 0.6

    dataset_path = '/scratches/nazgul_2/as2562/datasets/3DPW/test'
    dataset = PW3DEvalDataset(dataset_path,
                              img_wh=model_cfg.MODEL.IMAGE_SIZE,
                              selected_fnames=selected_fnames,
                              visible_joints_threshold=vis_joints_threshold)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pve', 'pve_sc', 'pve_pa', 'pve-t', 'pve-t_sc', 'mpjpe', 'mpjpe_sc', 'mpjpe_pa']
    metrics.extend([metric + '_samples_min' for metric in metrics ])
    metrics.extend(['verts_samples_dist_from_mean', 'joints3D_coco_samples_dist_from_mean', 'joints3D_coco_invis_samples_dist_from_mean'])
    metrics.append('hrnet_joints2Dsamples_l2es')

    save_path = '/scratch/as2562/ProHMR/evaluations/3dpw'
    if selected_fnames is not None:
        save_path += '_selected_fnames_occluded_joints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run evaluation
    evaluate_3dpw(model=model,
                  eval_dataset=dataset,
                  metrics=metrics,
                  device=device,
                  vis_save_path=save_path,
                  num_pred_samples=args.num_samples,
                  num_workers=4,
                  pin_memory=True,
                  vis_every_n_batches=vis_every_n_batches)







