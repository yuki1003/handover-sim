# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import easysim

from yacs.config import CfgNode as CN
import yaml

from handover.config import cfg


_C = cfg

# ---------------------------------------------------------------------------- #
# Simulation config
# ---------------------------------------------------------------------------- #
_C.SIM.RENDER = True
_C.SIM.TIME_STEP = 0.001

# ---------------------------------------------------------------------------- #
# Environment config
# ---------------------------------------------------------------------------- #
_C.ENV.ID = "HandoverPerActStateEnv-v1"
_C.ENV.TABLE_BASE_POSITION = (0.61, 0.0, 0.0) #(0.61, 0.28, 0.0)
_C.ENV.TABLE_BASE_ORIENTATION = (0.0, 0.0, 0.7071068, 0.7071068) #(0, 0, 0, 1)
_C.ENV.PANDA_INITIAL_POSITION = (0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)
_C.ENV.YCB_LOAD_MODE = "grasp_only"#"all"# 
_C.ENV.RENDERER_CAMERA_WIDTH = 128 #1600
_C.ENV.RENDERER_CAMERA_HEIGHT = 128 #900
_C.ENV.RENDERER_CAMERA_VERTICAL_FOV = 60 #60.0
_C.ENV.RENDERER_CAMERA_NEAR = 0.1 #NOTE: Starting the render
_C.ENV.RENDERER_CAMERA_FAR = 4 #10 #NOTE:How far the render is (Cut-off point)
_C.ENV.PERACT_RENDERER_CAMERA_SCENE_AMOUNT = 3 # Create radial number of cameras
_C.ENV.PERACT_RENDERER_CAMERA_SCENE_DISTANCE_HOR = 1 # Horizontal distance w.r.t. goal
_C.ENV.PERACT_RENDERER_CAMERA_SCENE_DISTANCE_VER = 0.2 # Vertical distance w.r.t. goal

# ---------------------------------------------------------------------------- #
# Benchmark config
# ---------------------------------------------------------------------------- #
_C.BENCHMARK.SETUP = "s1"
_C.BENCHMARK.SPLIT = "train"
_C.BENCHMARK.HANDOVER_OBJECTS = ["banana", "power_drill"]

_C.BENCHMARK.MAX_EPISODE_TIME = 10.0
_C.BENCHMARK.DRAW_GOAL = False
_C.BENCHMARK.RENDER_FRAME_RATE = 60 # camera feed rate - NOTE: PerAct Expert Demonstration recording collects data at 20 Hz
_C.BENCHMARK.TIME_WAIT = 3.0
_C.BENCHMARK.TIME_ACTION_REPEAT = 0.5

# ---------------------------------------------------------------------------- #
# AGENT config
# ---------------------------------------------------------------------------- #
_C.AGENT = CN()

# _C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/models/2024-11-29_04-23/best_model_test"#"/media/ywatabe/B4F2AA4FF2AA15A0/handoversim/outputs/models/handing_over_banana/2024-12-10_11-44/best_model_general"
_C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/outputs/models/handing_over_banana/2024-12-10_14-59/best_model_general"
# _C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/models/2024-11-29_04-23/best_model_train"
# _C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/outputs/models/handing_over_banana/2024-12-12_17-25/best_model_general" # {crop skip 10}
# _C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/outputs/models/handing_over_banana/2024-12-12_21-23/best_model_general" # {crop skip 5}
_C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/outputs/models/handing_over_banana/2024-12-16_11-25/best_model_general" # {crop skip 5 new}
# _C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/outputs/models/handing_over_banana/2024-12-13_14-46/best_model_test" # Check the approach
# _C.AGENT.model_path = "/home/ywatabe/Projects/PerAct/outputs/models/handing_over_banana/2024-12-26_16-29/best_model_train" # Check the approach new
_C.AGENT.language_goal = "handing over banana"

_C.AGENT.PERCEIVOR_IO = CN()
_C.AGENT.PERCEIVOR_IO.depth = 6
_C.AGENT.PERCEIVOR_IO.iterations = 1
_C.AGENT.PERCEIVOR_IO.voxel_size = 100
_C.AGENT.PERCEIVOR_IO.initial_dim = 3 + 3 + 1 + 3
_C.AGENT.PERCEIVOR_IO.low_dim_size = 4
_C.AGENT.PERCEIVOR_IO.layer = 0
_C.AGENT.PERCEIVOR_IO.num_rotation_classes = 72
_C.AGENT.PERCEIVOR_IO.num_grip_classes = 2
_C.AGENT.PERCEIVOR_IO.num_collision_classes = 2
_C.AGENT.PERCEIVOR_IO.num_latents = 512
_C.AGENT.PERCEIVOR_IO.latent_dim = 512
_C.AGENT.PERCEIVOR_IO.cross_heads = 1
_C.AGENT.PERCEIVOR_IO.latent_heads = 8
_C.AGENT.PERCEIVOR_IO.cross_dim_head = 64
_C.AGENT.PERCEIVOR_IO.latent_dim_head = 64
_C.AGENT.PERCEIVOR_IO.weight_tie_layers = False
_C.AGENT.PERCEIVOR_IO.activation = 'lrelu'
_C.AGENT.PERCEIVOR_IO.input_dropout = 0.1
_C.AGENT.PERCEIVOR_IO.attn_dropout = 0.1
_C.AGENT.PERCEIVOR_IO.decoder_dropout = 0.0
_C.AGENT.PERCEIVOR_IO.voxel_patch_size = 5
_C.AGENT.PERCEIVOR_IO.voxel_patch_stride = 5
_C.AGENT.PERCEIVOR_IO.final_dim = 64

_C.AGENT.PERACT = CN()
_C.AGENT.PERACT.coordinate_bounds = [0.11, -0.5, 0.8, 1.11, 0.5, 1.8]
_C.AGENT.PERACT.camera_names = ["view_0", "view_1", "view_2"]
_C.AGENT.PERACT.batch_size = 1
_C.AGENT.PERACT.voxel_size = _C.AGENT.PERCEIVOR_IO.voxel_size
_C.AGENT.PERACT.voxel_feature_size = 3
_C.AGENT.PERACT.num_rotation_classes = _C.AGENT.PERCEIVOR_IO.num_rotation_classes # Depends on Rotation resolution (i.e. 360/rotation_resolution)
_C.AGENT.PERACT.rotation_resolution = 5
_C.AGENT.PERACT.image_resolution = [128, 128]

cfg = _C
get_cfg = easysim.get_cfg

get_config_from_args = easysim.get_config_from_args
# _C.AGENT.PERACT.language_goal = "handing over {}".format()
# Convert CfgNode to dictionary
def cfg_to_dict(cfg_node):
    yaml_str = cfg_node.dump()  # Serialize to YAML string
    return yaml.safe_load(yaml_str)  # Parse YAML string back into a dictionary

# print(cfg_to_dict(cfg.AGENT.PERACT))