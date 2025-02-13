"""
Parts of the code are taken or adapted from
https://github.com/mkocabas/EpipolarPose/blob/master/lib/utils/img_utils.py
"""
import torch
import numpy as np
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode

def do_augmentation(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        scale (float): Box rescaling factor.
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis. 
    """

    tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.SCALE_FACTOR + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * aug_config.ROT_FACTOR if random.random() <= aug_config.ROT_AUG_RATE else 0
    do_flip = aug_config.DO_FLIP and random.random() <= aug_config.FLIP_AUG_RATE
    do_extreme_crop = random.random() <= aug_config.EXTREME_CROP_AUG_RATE
    c_up = 1.0 + aug_config.COLOR_SCALE
    c_low = 1.0 - aug_config.COLOR_SCALE
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty

def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d: np.array, trans: np.array):
    """
    Transform a 2D point using translation matrix trans.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        trans (np.array): Transformation matrix.
    Returns:
        np.array: Transformed 2D point.
    """
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def generate_image_patch(img: np.array, c_x: float, c_y: float,
                         bb_width: float, bb_height: float,
                         patch_width: float, patch_height: float,
                         do_flip: bool, scale: float, rot: float) -> Tuple[np.array, np.array]:
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1


    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans


def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img

def fliplr_params(smpl_params: Dict, has_smpl_params: Dict) -> Tuple[Dict, Dict]:
    """
    Flip SMPL parameters when flipping the image.
    Args:
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
    Returns:
        Dict, Dict: Flipped SMPL parameters and valid flags.
    """
    global_orient = smpl_params['global_orient'].copy()
    body_pose = smpl_params['body_pose'].copy()
    betas = smpl_params['betas'].copy()
    has_global_orient = has_smpl_params['global_orient'].copy()
    has_body_pose = has_smpl_params['body_pose'].copy()
    has_betas = has_smpl_params['betas'].copy()

    body_pose_permutation = [6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
                             14 ,18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
                             34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
                             45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
                             56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
    body_pose_permutation = body_pose_permutation[:len(body_pose)]
    body_pose_permutation = [i-3 for i in body_pose_permutation]

    body_pose = body_pose[body_pose_permutation]

    global_orient[1::3] *= -1
    global_orient[2::3] *= -1
    body_pose[1::3] *= -1
    body_pose[2::3] *= -1

    smpl_params = {'global_orient': global_orient.astype(np.float32),
                   'body_pose': body_pose.astype(np.float32),
                   'betas': betas.astype(np.float32)
                  }

    has_smpl_params = {'global_orient': has_global_orient,
                       'body_pose': has_body_pose,
                       'betas': has_betas
                      }

    return smpl_params, has_smpl_params


def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]) -> np.array:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]

    return joints

def keypoint_3d_processing(keypoints_3d: np.array, flip_permutation: List[int], rot: float, do_flip: float) -> np.array:
    """
    Process 3D keypoints (rotation/flipping).
    Args:
        keypoints_3d (np.array): Input array of shape (N, 4) containing the 3D keypoints and confidence.
        flip_permutation (List): Permutation to apply after flipping.
        rot (float): Random rotation applied to the keypoints.
        do_flip (bool): Whether to flip keypoints or not.
    Returns:
        np.array: Transformed 3D keypoints with shape (N, 4).
    """
    if do_flip:
        keypoints_3d = fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    # in-plane rotation
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = -rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
    keypoints_3d[:, :-1] = np.einsum('ij,kj->ki', rot_mat, keypoints_3d[:, :-1])
    # flip the x coordinates
    keypoints_3d = keypoints_3d.astype('float32')
    return keypoints_3d

def rot_aa(aa: np.array, rot: float) -> np.array:
    """
    Rotate axis angle parameters.
    Args:
        aa (np.array): Axis-angle vector of shape (3,).
        rot (np.array): Rotation angle in degrees.
    Returns:
        np.array: Rotated axis-angle vector.
    """
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa.astype(np.float32)

def smpl_param_processing(smpl_params: Dict, has_smpl_params: Dict, rot: float, do_flip: bool) -> Tuple[Dict, Dict]:
    """
    Apply random augmentations to the SMPL parameters.
    Args:
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
        rot (float): Random rotation applied to the keypoints.
        do_flip (bool): Whether to flip keypoints or not.
    Returns:
        Dict, Dict: Transformed SMPL parameters and valid flags.
    """
    if do_flip:
        smpl_params, has_smpl_params = fliplr_params(smpl_params, has_smpl_params)
    smpl_params['global_orient'] = rot_aa(smpl_params['global_orient'], rot)
    return smpl_params, has_smpl_params



def get_example(img_path: str, center_x: float, center_y: float,
                width: float, height: float,
                keypoints_2d: np.array, keypoints_3d: np.array,
                smpl_params: Dict, has_smpl_params: Dict,
                flip_kp_permutation: List[int],
                patch_width: int, patch_height: int,
                mean: np.array, std: np.array,
                do_augment: bool, augm_config: CfgNode) -> Tuple:
    """
    Get an example from the dataset and (possibly) apply random augmentations.
    Args:
        img_path (str): Image filename
        center_x (float): Bounding box center x coordinate in the original image.
        center_y (float): Bounding box center y coordinate in the original image.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array with shape (N,3) containing the 2D keypoints in the original image coordinates.
        keypoints_3d (np.array): Array with shape (N,4) containing the 3D keypoints.
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
        flip_kp_permutation (List): Permutation to apply to the keypoints after flipping.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        mean (np.array): Array of shape (3,) containing the mean for normalizing the input image.
        std (np.array): Array of shape (3,) containing the std for normalizing the input image.
        do_augment (bool): Whether to apply data augmentation or not.
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size
        img_patch (np.array): Cropped image patch of shape (3, patch_height, patch_height)
        keypoints_2d (np.array): Array with shape (N,3) containing the transformed 2D keypoints.
        keypoints_3d (np.array): Array with shape (N,4) containing the transformed 3D keypoints.
        smpl_params (Dict): Transformed SMPL parameters.
        has_smpl_params (Dict): Valid flag for transformed SMPL parameters.
        img_size (np.array): Image size of the original image.
        """
    # 1. load image
    cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % img_path)
    img_height, img_width, img_channels = cvimg.shape

    img_size = np.array([img_height, img_width])

    # 2. get augmentation params
    if do_augment:
        scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = do_augmentation(augm_config)
    else:
        scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = 1.0, 0, False, False, [1.0, 1.0, 1.0], 0., 0.

    if do_extreme_crop:
        center_x, center_y, width, height = extreme_cropping(center_x, center_y, width, height, keypoints_2d)
    center_x += width * tx
    center_y += height * ty

    # Process 3D keypoints
    keypoints_3d = keypoint_3d_processing(keypoints_3d, flip_kp_permutation, rot, do_flip)

    # 3. generate image patch
    img_patch_cv, trans = generate_image_patch(cvimg,
                                               center_x, center_y,
                                               width, height,
                                               patch_width, patch_height,
                                               do_flip, scale, rot)

    image = img_patch_cv.copy()
    image = image[:, :, ::-1]
    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)


    smpl_params, has_smpl_params = smpl_param_processing(smpl_params, has_smpl_params, rot, do_flip)

    # apply normalization
    for n_c in range(img_channels):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    if do_flip:
        keypoints_2d = fliplr_keypoints(keypoints_2d, img_width, flip_kp_permutation)


    for n_jt in range(len(keypoints_2d)):
        keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
    keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5

    return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size

def crop_to_hips(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25+0, 25+1, 25+4, 25+5]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_to_shoulders(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    center, scale = get_bbox(keypoints_2d)
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height

def crop_to_head(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height

def full_body(keypoints_2d: np.array) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """

    body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_keypoints = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    return (np.maximum(keypoints_2d[body_keypoints, -1], keypoints_2d[body_keypoints_openpose, -1]) > 0).sum() == len(body_keypoints)

def upper_body(keypoints_2d: np.array):
    """
    Check if all upper body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    lower_body_keypoints_openpose = [10, 11, 13, 14]
    lower_body_keypoints = [25 + i for i in [1, 0, 4, 5]]
    upper_body_keypoints_openpose = [0, 1, 15, 16, 17, 18]
    upper_body_keypoints = [25+8, 25+9, 25+12, 25+13, 25+17, 25+18]
    return ((keypoints_2d[lower_body_keypoints + lower_body_keypoints_openpose, -1] > 0).sum() == 0)\
       and ((keypoints_2d[upper_body_keypoints + upper_body_keypoints_openpose, -1] > 0).sum() >= 2)

def get_bbox(keypoints_2d: np.array, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from openpose detections.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.array): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:,-1] > 0
    valid_keypoints = keypoints_2d[valid][:,:-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale

def extreme_cropping(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)

    return center_x, center_y, max(width, height), max(width, height)


def batch_crop_opencv_affine(output_wh,
                             num_to_crop,
                             iuv=None,
                             joints2D=None,
                             rgb=None,
                             seg=None,
                             bbox_centres=None,
                             bbox_heights=None,
                             bbox_widths=None,
                             bbox_whs=None,
                             joints2D_vis=None,
                             orig_scale_factor=1.2,
                             delta_scale_range=None,
                             delta_centre_range=None,
                             out_of_frame_pad_val=0,
                             solve_for_affine_trans=False,
                             uncrop=False,
                             uncrop_wh=None):
    """
    :param output_wh: tuple, output image (width, height)
    :param num_to_crop: scalar int, number of images in batch
    :param iuv: (B, 3, H, W)
    :param joints2D: (B, K, 2)
    :param rgb: (B, 3, H, W)
    :param seg: (B, H, W)
    :param bbox_centres: (B, 2), bounding box centres in (vertical, horizontal) coordinates
    :param bbox_heights: (B,)
    :param bbox_widths: (B,)
    :param bbox_whs: (B,) width/height for square bounding boxes
    :param joints2D_vis: (B, K)
    :param orig_scale_factor: original bbox scale factor (pre-augmentation)
    :param delta_scale_range: bbox scale augmentation range
    :param delta_centre_range: bbox centre augmentation range
    :param out_of_frame_pad_val: padding value for out-of-frame region after affine transform
    :param solve_for_affine_trans: bool, if true use cv2.getAffineTransform() to determine
        affine transformation matrix.
    :param uncrop: bool, if true uncrop image by applying inverse affine transformation
    :param uncrop_wh: tuple, output image size for uncropping.
    :return: cropped iuv/joints2D/rgb/seg, resized to output_wh
    """
    output_wh = np.array(output_wh, dtype=np.float32)
    cropped_dict = {}
    if iuv is not None:
        if not uncrop:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if joints2D is not None:
        cropped_dict['joints2D'] = np.zeros_like(joints2D)
    if rgb is not None:
        if not uncrop:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if seg is not None:
        if not uncrop:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)

    for i in range(num_to_crop):
        if bbox_centres is None:
            assert (iuv is not None) or (joints2D is not None) or (seg is not None), "Need either IUV, Seg or 2D Joints to determine bounding boxes!"
            if iuv is not None:
                # Determine bounding box corners from segmentation foreground/body pixels from IUV map
                body_pixels = np.argwhere(iuv[i, 0, :, :] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif seg is not None:
                # Determine bounding box corners from segmentation foreground/body pixels
                body_pixels = np.argwhere(seg[i] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif joints2D is not None:
                # Determine bounding box corners from 2D joints
                visible_joints2D = joints2D[i, joints2D_vis[i]]
                bbox_corners = np.concatenate([np.amin(visible_joints2D, axis=0)[::-1],   # (hor, vert) coords to (vert, hor) coords
                                               np.amax(visible_joints2D, axis=0)[::-1]])
                if (bbox_corners[:2] == bbox_corners[2:]).all():  # This can happen if only 1 joint is visible in input
                    print('Only 1 visible joint in input!')
                    bbox_corners[2:] = bbox_corners[:2] + output_wh / 4.
            bbox_centre, bbox_height, bbox_width = convert_bbox_corners_to_centre_hw(bbox_corners)
        else:
            bbox_centre = bbox_centres[i]
            if bbox_whs is not None:
                bbox_height = bbox_whs[i]
                bbox_width = bbox_whs[i]
            else:
                bbox_height = bbox_heights[i]
                bbox_width = bbox_widths[i]

        if not uncrop:
            # Change bounding box aspect ratio to match output aspect ratio
            aspect_ratio = output_wh[1] / output_wh[0]
            if bbox_height > bbox_width * aspect_ratio:
                bbox_width = bbox_height / aspect_ratio
            elif bbox_height < bbox_width * aspect_ratio:
                bbox_height = bbox_width * aspect_ratio

            # Scale bounding boxes + Apply random augmentations
            if delta_scale_range is not None:
                l, h = delta_scale_range
                delta_scale = (h - l) * np.random.rand() + l
                scale_factor = orig_scale_factor + delta_scale
            else:
                scale_factor = orig_scale_factor
            bbox_height = bbox_height * scale_factor
            bbox_width = bbox_width * scale_factor
            if delta_centre_range is not None:
                l, h = delta_centre_range
                delta_centre = (h - l) * np.random.rand(2) + l
                bbox_centre = bbox_centre + delta_centre

            # Determine affine transformation mapping bounding box to output image
            output_centre = output_wh * 0.5
            if solve_for_affine_trans:
                # Solve for affine transformation using 3 point correspondences (6 equations)
                bbox_points = np.zeros((3, 2), dtype=np.float32)
                bbox_points[0, :] = bbox_centre[::-1]  # (vert, hor) coordinates to (hor, vert coordinates)
                bbox_points[1, :] = bbox_centre[::-1] + np.array([bbox_width * 0.5, 0], dtype=np.float32)
                bbox_points[2, :] = bbox_centre[::-1] + np.array([0, bbox_height * 0.5], dtype=np.float32)

                output_points = np.zeros((3, 2), dtype=np.float32)
                output_points[0, :] = output_centre
                output_points[1, :] = output_centre + np.array([output_wh[0] * 0.5, 0], dtype=np.float32)
                output_points[2, :] = output_centre + np.array([0, output_wh[1] * 0.5], dtype=np.float32)
                affine_trans = cv2.getAffineTransform(bbox_points, output_points)
            else:
                # Hand-code affine transformation matrix - easy for cropping = scale + translate
                affine_trans = np.zeros((2, 3), dtype=np.float32)
                affine_trans[0, 0] = output_wh[0] / bbox_width
                affine_trans[1, 1] = output_wh[1] / bbox_height
                affine_trans[:, 2] = output_centre - (output_wh / np.array([bbox_width, bbox_height])) * bbox_centre[::-1]  # (vert, hor) coords to (hor, vert) coords
        else:
            # Hand-code inverse affine transformation matrix - easy for UN-cropping = scale + translate
            affine_trans = np.zeros((2, 3), dtype=np.float32)
            output_centre = output_wh * 0.5
            affine_trans[0, 0] = bbox_width / output_wh[0]
            affine_trans[1, 1] = bbox_height / output_wh[1]
            affine_trans[:, 2] = bbox_centre[::-1] - (np.array([bbox_width, bbox_height]) / output_wh) * output_centre

        # Apply affine transformation inputs.
        if iuv is not None:
            cropped_dict['iuv'][i] = cv2.warpAffine(src=iuv[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=out_of_frame_pad_val).transpose(2, 0, 1)
        if joints2D is not None:
            joints2D_homo = np.concatenate([joints2D[i], np.ones((joints2D.shape[1], 1))],
                                           axis=-1)
            cropped_dict['joints2D'][i] = np.einsum('ij,kj->ki', affine_trans, joints2D_homo)

        if rgb is not None:
            cropped_dict['rgb'][i] = cv2.warpAffine(src=rgb[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0).transpose(2, 0, 1)
        if seg is not None:
            cropped_dict['seg'][i] = cv2.warpAffine(src=seg[i],
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0)

    return cropped_dict