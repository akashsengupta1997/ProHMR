# ------------------------------------ Paths ------------------------------------
SMPL_MODEL_DIR = 'data/smpl'
J_REGRESSOR_EXTRA = 'data/J_regressor_extra.npy'
COCOPLUS_REGRESSOR = 'data/cocoplus_regressor.npy'
H36M_REGRESSOR = 'data/J_regressor_h36m.npy'
# ------------------------------------ Constants ------------------------------------
#FOCAL_LENGTH = 5000.
FOCAL_LENGTH = 300.

# ------------------------------------ Measurements ------------------------------------
# Metail selected measurement names (from SS measurement service)
METAIL_MEASUREMENTS = ['Chest', 'Left Bicep', 'Right Bicep', 'Left Forearm',
                       'Right Forearm', 'Stomach', 'Waist (men)', 'Waist (women)',
                       'Seat', 'Left Thigh', 'Right Thigh', 'Left Calf', 'Right Calf', 'Subject Height']

# SMPL measurements - defined by me, uses SMPL vertex IDs and joint IDs as measurement end-point definitions.
JOINT_LENGTH_MEAS_NAMES = ['Torso_Length',
                           'L_Thigh_Length', 'R_Thigh_Length', 'L_Calf_Length', 'R_Calf_Length',
                           # 'L_Upper_Arm_Length', 'R_Upper_Arm_Length', 'L_Forearm_Length', 'R_Forearm_Length',
                           'L_Arm_Length', 'R_Arm_Length',
                           # 'Shoulder_Width'
                           ]
JOINT_LENGTH_MEAS_INDEXES = [[0, 15],   # Hip to Top of Neck = Torso
                             [1, 4],    # L Hip to L Knee = L Thigh
                             [2, 5],    # R Hip to R Knee = R Thigh
                             [4, 7],    # L Knee to L Ankle = L Calf
                             [5, 8],    # R Knee to R Ankle = R Calf
                             [16, 20],  # L Shoulder to L Wrist = L Arm
                             [17, 21]  # R Shoulder to R Wrist = R Arm
                             ]
VERTEX_CIRCUMFERENCE_MEAS_NAMES = ['L_Forearm_Circum', 'R_Forearm_Circum', 'L_Upper_Arm_Circum', 'R_Upper_Arm_Circum',
                                   'L_Calf_Circum', 'R_Calf_Circum', 'L_Lower_Thigh_Circum', 'R_Lower_Thigh_Circum',
                                   'L_Upper_Thigh_Circum', 'R_Upper_Thigh_Circum', 'Neck_Circum']
VERTEX_CIRCUMFERENCE_MEAS_INDEXES = [[1557, 1558, 1587, 1554, 1553, 1727, 1583, 1584, 1689, 1687, 1686, 1590, 1591, 1548, 1547, 1551],  # Left Forearm
                                     [5027, 5028, 5020, 5018, 5017, 5060, 5061, 5157, 5156, 5159, 5053, 5054, 5196, 5024, 5023, 5057],  # Right Forearm
                                     [628, 627, 789, 1311, 1315, 1379, 1378, 1394, 1393, 1389, 1388, 1233, 1232, 1385, 1381, 1382, 1397, 1396],  # Left Upper Arm
                                     [4117, 4277, 4791, 4794, 4850, 4851, 4865, 4866, 4862, 4863, 4716, 4717, 4859, 4856, 4855, 4870, 4871, 4116],  # Right Upper Arm
                                     [1074, 1077, 1470, 1094, 1095, 1473, 1465, 1466, 1108, 1111, 1530, 1089, 1086, 1154, 1372],  # Left Calf
                                     [4583, 4580, 4943, 4561, 4560, 4845, 4640, 4572, 4573, 5000, 4595, 4594, 4940, 4938, 4946],  # Right Calf
                                     [1041, 1147, 1171, 1172, 1029, 1030, 1167, 1033, 1034, 1035, 1037, 1036, 1038, 1040, 1039, 1520, 1042],  # Left Lower Thigh
                                     [4528, 4632, 4657, 4660, 4515, 4518, 4653, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4991, 4527],  # Right Lower Thigh
                                     [910, 1365, 907, 906, 957, 904, 905, 903, 901, 962, 898, 899, 934, 935, 1453, 964, 909, 910],  # Left Upper Thigh
                                     [4397, 4396, 4838, 4393, 4392, 4443, 4391, 4390, 4388, 4387, 4448, 4386, 4385, 4422, 4421, 4926, 4449],  # Right Upper Thigh
                                     # [3165, 188, 374, 178, 457, 3690, 5240, 3700],  # Head
                                     [3050, 3839, 3796, 3797, 3662, 3663, 3810, 3718, 3719, 3723, 3724, 3768, 3918, 460, 423, 257, 212, 213, 209, 206, 298, 153, 150, 285, 284, 334]]  # Neck
VERTEX_LENGTH_MEAS_NAMES = ['Chest_Width', 'Stomach_Width', 'Hip_Width', 'Abdomen_Width',
                            'Chest_Depth', 'Stomach_Depth', 'Hip_Depth', 'Abdomen_Depth',
                            'Head_Width', 'Head_Height', 'Head_Depth',
                            'Underbust_Depth',
                            'Shoulder_Width',
                            # 'L_Upper_Arm_Length', 'R_Upper_Arm_Length', 'L_Forearm_Length', 'R_Forearm_Length'
                            ]
VERTEX_LENGTH_MEAS_INDEXES = [[4226, 738],   # Chest Width
                              [4804, 1323],  # Waist/Stomach Width
                              # [4920, 1446],  # Hip Width (higher)
                              [6550, 3129],  # Hip Width
                              [1794, 5256],  # Abdomen Width
                              [3076, 3015],  # Chest Depth
                              [3509, 3502],  # Waist/Stomach Depth
                              # [3119, 1807],  # Hip Depth (higher)
                              [3145, 3141],  # Hip Depth
                              [3507, 3159],  # Abdomen Depth
                              [368, 3872],  # Head Width
                              [412, 3058],  # Head Height
                              [457, 3165],  # Head Depth
                              [1329, 3017],  # Underbust Depth
                              [1509, 4982],  # Shoulder Width
                              # [1509, 1651],  # Left Upper Arm Length
                              # [4982, 5120],  # Right Upper Arm Length
                              # [1651, 2230],  # Left Forearm Length
                              # [5120, 5691]   # Right Forearm Length
                              ]
ALL_MEAS_NAMES = JOINT_LENGTH_MEAS_NAMES + VERTEX_LENGTH_MEAS_NAMES + VERTEX_CIRCUMFERENCE_MEAS_NAMES
ALL_MEAS_NAMES_NO_SYMM = [name.replace('L_', '') for name in ALL_MEAS_NAMES if not name.startswith('R_')]
NUMBER_OF_MEAS_TYPES = len(ALL_MEAS_NAMES)
NUMBER_OF_MEAS_TYPES_NO_SYMM = len(ALL_MEAS_NAMES_NO_SYMM)

# ------------------------------------ Joint label conventions ------------------------------------
# These are the 2D joints output from the SMPL model in the smpl_from_lib file
# (with the extra joint regressor)
SMPL_JOINTS = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
# These are the 2D joints output from KeyPoint-RCNN in detectron2 (trained on COCO - basically
# these are the COCO joint annotations)
# Note that these are different to the cocoplus joints output by the cocoplus regressor.
COCO_JOINTS = {
    'Right Ankle': 16, 'Right Knee': 14, 'Right Hip': 12,
    'Left Hip': 11, 'Left Knee': 13, 'Left Ankle': 15,
    'Right Wrist': 10, 'Right Elbow': 8, 'Right Shoulder': 6,
    'Left Shoulder': 5, 'Left Elbow': 7, 'Left Wrist': 9,
    'Right Ear': 4, 'Left Ear': 3, 'Right Eye': 2, 'Left Eye': 1,
    'Nose': 0
}

# The SMPL model (im smpl_official.py) returns a large superset of joints.
# Different subsets are used during training - e.g. H36M 3D joints convention and COCO 2D joints convention.

# The joints superset is broken down into: 45 SMPL joints, 9 extra joints, 19 cocoplus joints
# and 17 H36M joints. The 45 SMPL joints are converted to 17 COCO joints with the map below.
# (Not really sure how coco and cocoplus are related)

# Joint label conversions
ALL_JOINTS_TO_COCO_MAP = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]  # Using OP Hips
ALL_JOINTS_TO_H36M_MAP = list(range(73, 90))

# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

# SMPL parameters corresponding to COCO joints [5: L shoulder, 6: R shoulder, 7: L elbow, 8: R elbow, 11: L hip, 12: R hip, 13: L knee, 14: R knee]
COCO_TO_SMPL_LIMB_POSE_PARAMS = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]
COCO_TO_SMPL_LIMB_POSE_PARAMS_WITH_GLOB = [i + 3 for i in COCO_TO_SMPL_LIMB_POSE_PARAMS]   # Add 3 when including global pose params as elements 0, 1, 2.

# Joint label and body part seg label matching
# COCO Joints: 14 part seg
COCO_JOINTS_TO_FOURTEEN_PART_SEG_MAP = {7: 3,
                                        8: 5,
                                        9: 12,
                                        10: 11,
                                        13: 7,
                                        14: 9,
                                        15: 14,
                                        16: 13}
# 14 part seg: COCO Joints
FOURTEEN_PART_SEG_TO_COCO_JOINTS_MAP = {3: 7,
                                        5: 8,
                                        12: 9,
                                        11: 10,
                                        7: 13,
                                        9: 14,
                                        14: 15,
                                        13: 16}
# 24 part seg: COCO Joints
TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP = {19: 7,
                                          21: 7,
                                          20: 8,
                                          22: 8,
                                          4: 9,
                                          3: 10,
                                          12: 13,
                                          14: 13,
                                          11: 14,
                                          13: 14,
                                          5: 15,
                                          6: 16}

