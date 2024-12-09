# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import easysim

from yacs.config import CfgNode as CN

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

# RLBench start
# _C.ENV.PANDA_INITIAL_POSITION = (3.5779521567746997e-09, 0.1745329201221466, 3.305009599330333e-08, -0.8726646304130554, -1.1409618139168742e-07, 1.2217304706573486, 0.7853981256484985, 0.04, 0.04)

# Handoversim start
_C.ENV.PANDA_INITIAL_POSITION = (0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)

_C.ENV.YCB_LOAD_MODE = "all" # "grasp_only"

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

_C.BENCHMARK.SETUP = "s0"

_C.BENCHMARK.SPLIT = "train"

_C.BENCHMARK.HANDOVER_OBJECTS = ["banana"]

_C.BENCHMARK.DRAW_GOAL = False

_C.BENCHMARK.RECORD = False

_C.BENCHMARK.RECORD_DIR = "task_data/handoversim/{}_{}".format(_C.BENCHMARK.SPLIT, _C.BENCHMARK.SETUP)

_C.BENCHMARK.RENDER_FRAME_RATE = 60 # camera feed rate - NOTE: PerAct Expert Demonstration recording collects data at 20 Hz

_C.BENCHMARK.TIME_WAIT = 3.0

_C.BENCHMARK.TIME_ACTION_REPEAT = 1.0 / _C.BENCHMARK.RENDER_FRAME_RATE

cfg = _C

get_cfg = easysim.get_cfg

get_config_from_args = easysim.get_config_from_args