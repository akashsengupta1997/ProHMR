import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import math

import my_config

from prohmr.configs import get_config, prohmr_config
from prohmr.models import ProHMR
from prohmr.models.smpl_mine import SMPL
from prohmr.utils.pose_utils import compute_similarity_transform_batch_numpy, scale_and_translation_transform_batch, check_joints2d_visibility_torch
from prohmr.utils.geometry import undo_keypoint_normalisation, orthographic_project_torch, convert_weak_perspective_to_camera_translation
from prohmr.utils.sampling_utils import compute_vertex_uncertainties_from_samples
from prohmr.utils.renderer import Renderer
from prohmr.datasets.ssp3d_eval_dataset import SSP3DEvalDataset


def evaluate_single_in_multitasknet_ssp3d(model,
                                          model_cfg,
                                          eval_dataset,
                                          metrics_to_track,
                                          device,
                                          save_path,
                                          num_pred_samples,
                                          num_workers=4,
                                          pin_memory=True,
                                          vis_every_n_batches=1,
                                          num_samples_to_visualise=10,
                                          save_per_frame_uncertainty=True,
                                          occlude=None,
                                          extreme_crop=False):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl_neutral = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1).to(device)
    smpl_male = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1, gender='male').to(device)
    smpl_female = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1, gender='female').to(device)

    metric_sums = {'num_datapoints': 0}
    per_frame_metrics = {}
    for metric in metrics_to_track:
        metric_sums[metric] = 0.
        per_frame_metrics[metric] = []

        if metric == 'joints3D_coco_invis_samples_dist_from_mean':
            metric_sums['num_invis_joints3Dsamples'] = 0

        elif metric == 'joints2D_l2es':
            metric_sums['num_vis_joints2D'] = 0
        elif metric == 'joints2Dsamples_l2es':
            metric_sums['num_vis_joints2Dsamples'] = 0

        elif metric == 'silhouette_ious':
            metric_sums['num_true_positives'] = 0.
            metric_sums['num_false_positives'] = 0.
            metric_sums['num_true_negatives'] = 0.
            metric_sums['num_false_negatives'] = 0.
        elif metric == 'silhouettesamples_ious':
            metric_sums['num_samples_true_positives'] = 0.
            metric_sums['num_samples_false_positives'] = 0.
            metric_sums['num_samples_true_negatives'] = 0.
            metric_sums['num_samples_false_negatives'] = 0.

    fname_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []
    if save_per_frame_uncertainty:
        vertices_uncertainty_per_frame = []

    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    reposed_cam_t = convert_weak_perspective_to_camera_translation(cam_wp=np.array([0.85, 0., -0.2]),
                                                                   focal_length=model_cfg.EXTRA.FOCAL_LENGTH,
                                                                   resolution=model_cfg.MODEL.IMAGE_SIZE)
    if extreme_crop:
        rot_cam_t = convert_weak_perspective_to_camera_translation(cam_wp=np.array([0.85, 0., 0.]),
                                                                   focal_length=model_cfg.EXTRA.FOCAL_LENGTH,
                                                                   resolution=model_cfg.MODEL.IMAGE_SIZE)

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input'].to(device)
        if occlude == 'bottom':
            input[:, :, input.shape[2] // 2:, :] = 0.
        elif occlude == 'side':
            input[:, :, :, input.shape[3] // 2:] = 0.

        target_shape = samples_batch['shape'].to(device)
        target_pose = samples_batch['pose'].to(device)
        target_silhouette = samples_batch['silhouette']
        target_joints2D_coco = samples_batch['keypoints']
        target_joints2D_coco_vis = check_joints2d_visibility_torch(target_joints2D_coco,
                                                                   input.shape[-1])  # (batch_size, 17)
        target_gender = samples_batch['gender'][0]

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
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # ------------------------------------------------ PREDICTIONS ------------------------------------------------
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

        pred_joints_coco_samples = pred_smpl_output_samples.joints[:, my_config.ALL_JOINTS_TO_COCO_MAP, :]  # (num_pred_samples, 17, 3)
        pred_joints2D_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)  # (num_pred_samples, 17, 2)
        pred_joints2D_coco_samples = undo_keypoint_normalisation(pred_joints2D_coco_samples, input.shape[-1])

        pred_reposed_vertices_samples = smpl_neutral(body_pose=torch.zeros(num_pred_samples, 69, device=device, dtype=torch.float32),
                                                     global_orient=torch.zeros(num_pred_samples, 3, device=device, dtype=torch.float32),
                                                     betas=pred_shape_samples).vertices  # (num_pred_samples, 6890, 3)

        # ------------------------------------------------ METRICS ------------------------------------------------

        # Numpy-fying targets
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()
        target_joints2D_coco = target_joints2D_coco.cpu().detach().numpy()
        target_joints2D_coco_vis = target_joints2D_coco_vis.cpu().detach().numpy()
        target_silhouette = target_silhouette.cpu().detach().numpy()

        # Numpy-fying preds
        pred_vertices_mode = pred_vertices_mode.cpu().detach().numpy()
        pred_joints_coco_mode = pred_joints_coco_mode.cpu().detach().numpy()
        pred_vertices2D_mode = pred_vertices2D_mode.cpu().detach().numpy()
        pred_joints2D_coco_mode = pred_joints2D_coco_mode.cpu().detach().numpy()
        pred_reposed_vertices_mean = pred_reposed_vertices_mean.cpu().detach().numpy()

        pred_vertices_samples = pred_vertices_samples.cpu().detach().numpy()
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
            target_joints2D_coco_invis = np.logical_not(target_joints2D_coco_vis[0])  # (17,)

            if np.any(target_joints2D_coco_invis):
                joints3D_coco_invis_samples = pred_joints_coco_samples[:, target_joints2D_coco_invis, :]  # (num samples, num invis joints, 3)
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
        if 'joints2D_l2es' in metrics_to_track:
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco_mode[:, target_joints2D_coco_vis[0], :] - target_joints2D_coco[:, target_joints2D_coco_vis[0], :],
                                                axis=-1)  # (1, num vis joints)
            assert joints2D_l2e_batch.shape[1] == target_joints2D_coco_vis.sum()

            metric_sums['joints2D_l2es'] += np.sum(joints2D_l2e_batch)  # scalar
            metric_sums['num_vis_joints2D'] += joints2D_l2e_batch.shape[1]
            per_frame_metrics['joints2D_l2es'].append(np.mean(joints2D_l2e_batch, axis=-1))  # (1,)

        if 'silhouette_ious' in metrics_to_track:
            _, pred_silhouette_mode = renderer(vertices=pred_vertices_mode[0],
                                               camera_translation=out['pred_cam_t'][0, 0, :].cpu().detach().numpy(),
                                               image=np.zeros((model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE, 3)),
                                               unnormalise_img=False,
                                               return_silhouette=True)
            pred_silhouette_mode = pred_silhouette_mode[None, :, :, 0].astype(np.float32)  # (1, img_wh, img_wh)
            true_positive = np.logical_and(pred_silhouette_mode, target_silhouette)
            false_positive = np.logical_and(pred_silhouette_mode, np.logical_not(target_silhouette))
            true_negative = np.logical_and(np.logical_not(pred_silhouette_mode), np.logical_not(target_silhouette))
            false_negative = np.logical_and(np.logical_not(pred_silhouette_mode), target_silhouette)
            num_tp = np.sum(true_positive, axis=(1, 2))  # (1,)
            num_fp = np.sum(false_positive, axis=(1, 2))
            num_tn = np.sum(true_negative, axis=(1, 2))
            num_fn = np.sum(false_negative, axis=(1, 2))
            metric_sums['num_true_positives'] += np.sum(num_tp)  # scalar
            metric_sums['num_false_positives'] += np.sum(num_fp)
            metric_sums['num_true_negatives'] += np.sum(num_tn)
            metric_sums['num_false_negatives'] += np.sum(num_fn)
            iou_per_frame = num_tp / (num_tp + num_fp + num_fn)
            per_frame_metrics['silhouette_ious'].append(iou_per_frame)  # (1,)
            # TODO check silhouette eval and add silhouette visualisation

        # -------------------------------- 2D Metrics after Averaging over Samples ---------------------------
        if 'joints2Dsamples_l2es' in metrics_to_track:
            joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples[:, target_joints2D_coco_vis[0], :] - target_joints2D_coco[:, target_joints2D_coco_vis[0], :],
                                                       axis=-1)  # (num_samples, num vis joints)
            assert joints2Dsamples_l2e_batch.shape[1] == target_joints2D_coco_vis.sum()

            metric_sums['joints2Dsamples_l2es'] += np.sum(joints2Dsamples_l2e_batch)  # scalar
            metric_sums['num_vis_joints2Dsamples'] += np.prod(joints2Dsamples_l2e_batch.shape)
            per_frame_metrics['joints2Dsamples_l2es'].append(np.mean(joints2Dsamples_l2e_batch)[None])  # (1,)

        if 'silhouettesamples_ious' in metrics_to_track:
            pred_silhouette_samples = []
            for i in range(num_pred_samples):
                _, silh_sample = renderer(vertices=pred_vertices_samples[i],
                                          camera_translation=out['pred_cam_t'][0, 0, :].cpu().detach().numpy(),
                                          image=np.zeros((model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE, 3)),
                                          unnormalise_img=False,
                                          return_silhouette=True)
                pred_silhouette_samples.append(silh_sample[:, :, 0].astype(np.float32))
            pred_silhouette_samples = np.stack(pred_silhouette_samples, axis=0)[None, :, :, :]  # (1, num_samples, img_wh, img_wh)
            target_silhouette_tiled = np.tile(target_silhouette[:, None, :, :], (1, num_pred_samples, 1, 1))  # (1, num_samples, img_wh, img_wh)
            print('HERE2', pred_silhouette_samples.shape, target_silhouette_tiled.shape,
                  pred_silhouette_samples.dtype, target_silhouette_tiled.dtype,
                  np.unique(pred_silhouette_samples), np.unique(target_silhouette_tiled))
            
            true_positive = np.logical_and(pred_silhouette_samples, target_silhouette_tiled)
            false_positive = np.logical_and(pred_silhouette_samples, np.logical_not(target_silhouette_tiled))
            true_negative = np.logical_and(np.logical_not(pred_silhouette_samples), np.logical_not(target_silhouette_tiled))
            false_negative = np.logical_and(np.logical_not(pred_silhouette_samples), target_silhouette_tiled)
            num_tp = np.sum(true_positive, axis=(1, 2, 3))  # (1,)
            num_fp = np.sum(false_positive, axis=(1, 2, 3))
            num_tn = np.sum(true_negative, axis=(1, 2, 3))
            num_fn = np.sum(false_negative, axis=(1, 2, 3))
            metric_sums['num_samples_true_positives'] += np.sum(num_tp)  # scalar
            metric_sums['num_samples_false_positives'] += np.sum(num_fp)
            metric_sums['num_samples_true_negatives'] += np.sum(num_tn)
            metric_sums['num_samples_false_negatives'] += np.sum(num_fn)
            iou_per_frame = num_tp / (num_tp + num_fp + num_fn)
            per_frame_metrics['silhouettesamples_ious'].append(iou_per_frame)  # (1,)

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
                                         camera_translation=pred_cam_t.copy(),  # copy() needed because renderer changes cam_t in place...
                                         image=vis_img[0],
                                         unnormalise_img=False)
            body_vis_rgb_mode_rot = renderer(vertices=pred_vertices_mode[0],
                                             camera_translation=pred_cam_t.copy() if not extreme_crop else rot_cam_t.copy(),
                                             image=np.zeros_like(vis_img[0]),
                                             unnormalise_img=False,
                                             angle=np.pi/2.,
                                             axis=[0., 1., 0.])

            reposed_body_vis_rgb_mean = renderer(vertices=pred_reposed_vertices_mean[0],
                                                 camera_translation=reposed_cam_t.copy(),
                                                 image=np.zeros_like(vis_img[0]),
                                                 unnormalise_img=False,
                                                 flip_updown=False)
            reposed_body_vis_rgb_mean_rot = renderer(vertices=pred_reposed_vertices_mean[0],
                                                     camera_translation=reposed_cam_t.copy(),
                                                     image=np.zeros_like(vis_img[0]),
                                                     unnormalise_img=False,
                                                     angle=np.pi / 2.,
                                                     axis=[0., 1., 0.],
                                                     flip_updown=False)

            body_vis_rgb_samples = []
            body_vis_rgb_rot_samples = []
            for i in range(num_samples_to_visualise):
                body_vis_rgb_samples.append(renderer(vertices=pred_vertices_samples[i],
                                                     camera_translation=pred_cam_t.copy(),
                                                     image=vis_img[0],
                                                     unnormalise_img=False))
                body_vis_rgb_rot_samples.append(renderer(vertices=pred_vertices_samples[i],
                                                         camera_translation=pred_cam_t.copy() if not extreme_crop else rot_cam_t.copy(),
                                                         image=np.zeros_like(vis_img[0]),
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

            # Plot silhouette+J2D and reposed body render
            if 'silhouette_ious' in metrics_to_track:
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(pred_silhouette_mode[0].astype(np.int16) - target_silhouette[0].astype(np.int16))
                plt.text(10, 10, s='mIOU: {:.4f}'.format(per_frame_metrics['silhouette_ious'][0]))
            if 'joints2D_l2es' in metrics_to_track:
                plt.scatter(target_joints2D_coco[0, :, 0],
                            target_joints2D_coco[0, :, 1],
                            c=target_joints2D_coco_vis[0, :].astype(np.float32), s=10.0)
                plt.scatter(pred_joints2D_coco_mode[0, :, 0],
                            pred_joints2D_coco_mode[0, :, 1],
                            c='r', s=10.0)
                for j in range(target_joints2D_coco.shape[1]):
                    plt.text(target_joints2D_coco[0, j, 0],
                             target_joints2D_coco[0, j, 1],
                             str(j))
                    plt.text(pred_joints2D_coco_mode[0, j, 0],
                             pred_joints2D_coco_mode[0, j, 1],
                             str(j))
                plt.text(10, 30, s='J2D L2E: {:.4f}'.format(per_frame_metrics['joints2D_l2es'][0]))
            subplot_count += 1

            if 'pves_sc' in metrics_to_track:
                # Plot PVE-SC pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
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
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 0],
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][batch_num][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][batch_num][0]))
                subplot_count += 1

            if 'pves_pa' in metrics_to_track:
                # Plot PVE-PA pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-PA')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
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
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 0],
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][batch_num][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][batch_num][0]))
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
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][batch_num][0]))
                subplot_count += 1

            # Plot per-vertex uncertainties
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.text(0.5, 0.5, s='Uncertainty for\nPVE')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
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
            plt.gca().invert_yaxis()
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
            plt.gca().invert_yaxis()
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
            plt.gca().invert_yaxis()
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
            num_subplots = num_samples_to_visualise * 2 + 2
            num_row = 4
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
            for i in range(num_samples_to_visualise):
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

        if metric_type == 'joints2D_l2es':
            joints2D_l2e = metric_sums['joints2D_l2es'] / metric_sums['num_vis_joints2D']
            final_metrics[metric_type] = joints2D_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_joints2D'])
        elif metric_type == 'joints2D_l2es_best_j2d_sample':
            joints2D_l2e_best_j2d_sample = metric_sums['joints2D_l2es_best_j2d_sample'] / metric_sums['num_vis_joints2D']
            final_metrics[metric_type] = joints2D_l2e_best_j2d_sample
        elif metric_type == 'joints2Dsamples_l2es':
            joints2Dsamples_l2e = metric_sums['joints2Dsamples_l2es'] / metric_sums['num_vis_joints2Dsamples']
            final_metrics[metric_type] = joints2Dsamples_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_joints2Dsamples'])

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
    parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
    parser.add_argument('--occlude', type=str, choices=['bottom', 'side'])
    parser.add_argument('--extreme_crop', '-EC', action='store_true')
    parser.add_argument('--extreme_crop_scale', '-ECS', type=float)
    parser.add_argument('--num_samples', '-N', type=int, default=25, help='Number of test samples to evaluate with')
    parser.add_argument('--vis_every', '-V', type=int, default=1)


    args = parser.parse_args()

    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)

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
    dataset_path = '/scratch/as2562/datasets/ssp_3d'
    dataset = SSP3DEvalDataset(dataset_path,
                               img_wh=model_cfg.MODEL.IMAGE_SIZE,
                               extreme_crop=args.extreme_crop,
                               extreme_crop_scale=args.extreme_crop_scale)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc']
    metrics.extend([metric + '_samples_min' for metric in metrics])
    metrics.extend(['verts_samples_dist_from_mean', 'joints3D_coco_samples_dist_from_mean', 'joints3D_coco_invis_samples_dist_from_mean'])
    metrics.append('joints2D_l2es')
    metrics.append('joints2Dsamples_l2es')
    metrics.append('silhouette_ious')
    metrics.append('silhouettesamples_ious')

    save_path = '/scratch/as2562/ProHMR/evaluations/ssp3d_{}_samples'.format(args.num_samples)
    if args.occlude is not None:
        save_path += '_occlude_{}'.format(args.occlude)
    if args.extreme_crop:
        save_path += '_extreme_crop_scale_{}'.format(args.extreme_crop_scale)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('Saving to:', save_path)

    # Run evaluation
    evaluate_single_in_multitasknet_ssp3d(model=model,
                                          model_cfg=model_cfg,
                                          eval_dataset=dataset,
                                          metrics_to_track=metrics,
                                          device=device,
                                          save_path=save_path,
                                          num_pred_samples=args.num_samples,
                                          num_workers=args.num_workers,
                                          pin_memory=True,
                                          vis_every_n_batches=args.vis_every,
                                          occlude=args.occlude,
                                          extreme_crop=args.extreme_crop,
                                          num_samples_to_visualise=10,
                                          save_per_frame_uncertainty=True)




