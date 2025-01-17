# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""
This is a benchmark recorder for Using RLBench format
We use this to train a PerActAgent
"""

import pickle
import shutil
import os
import functools
import time
import cv2
import easysim
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import gym

from handover.benchmark_runner import EpisodeStatus
from handover.benchmark_wrapper import HandoverStatusWrapper

from handover.ycb import YCB

import sys
sys.path.append("/home/bepgroup/Projects/PerAct_ws/peract_colab")
from rlbench.backend.observation import Observation
from rlbench.demo import Demo


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer


class HandoverBenchmarkWrapperFiltered(HandoverStatusWrapper):
    _EVAL_SKIP_OBJECT = [0, 15] # NOTE: Ungraspable objects by gripper

    _YCB_CLASSES = YCB.CLASSES
    
    def __init__(self, env):
        super().__init__(env)

        available_handover_objects = list(self._YCB_CLASSES.values())
        other_skip_objects = [i for i, string in enumerate(available_handover_objects) if not any(substring in string for substring in self.cfg.BENCHMARK.HANDOVER_OBJECTS)]
        self._EVAL_SKIP_OBJECT.extend(other_skip_objects)

        # Seen subjects, camera views, grasped objects.
        if self.cfg.BENCHMARK.SETUP == "s0":
            if self.cfg.BENCHMARK.SPLIT == "train":
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # NOTE: subject==person
                sequence_ind = [i for i in range(100) if i % 5 != 4] # NOTE:  right, right, left, left, (random)
            if self.cfg.BENCHMARK.SPLIT == "val":
                subject_ind = [0, 1]
                sequence_ind = [i for i in range(100) if i % 5 == 4] # NOTE:  (right, right, left, left), random
            if self.cfg.BENCHMARK.SPLIT == "test":
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                sequence_ind = [i for i in range(100) if i % 5 == 4] # NOTE:  (right, right, left, left), random
            mano_side = ["right", "left"]

        # Unseen subjects.
        elif self.cfg.BENCHMARK.SETUP == "s1":
            if self.cfg.BENCHMARK.SPLIT == "train":
                subject_ind = [0, 1, 2, 3, 4, 5, 9]
            if self.cfg.BENCHMARK.SPLIT == "val":
                subject_ind = [6]
            if self.cfg.BENCHMARK.SPLIT == "test":
                subject_ind = [7, 8]
            sequence_ind = [*range(100)]
            mano_side = ["right", "left"]

        # Unseen handedness. NOTE: Right vs. Left
        elif self.cfg.BENCHMARK.SETUP == "s2":
            if self.cfg.BENCHMARK.SPLIT == "train":
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                mano_side = ["right"]
            if self.cfg.BENCHMARK.SPLIT == "val":
                subject_ind = [0, 1]
                mano_side = ["left"]
            if self.cfg.BENCHMARK.SPLIT == "test":
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                mano_side = ["left"]
            sequence_ind = [*range(100)]

        # Unseen grasped objects.
        elif self.cfg.BENCHMARK.SETUP == "s3":
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            if self.cfg.BENCHMARK.SPLIT == "train":
                sequence_ind = [i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)]
            if self.cfg.BENCHMARK.SPLIT == "val":
                sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
            if self.cfg.BENCHMARK.SPLIT == "test":
                sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]
            mano_side = ["right", "left"]

        else:
            raise ValueError(f"Unrecognized 'cfg.BENCHMARK.SETUP' value: {self.cfg.BENCHMARK.SETUP}.")
        
        self._scene_ids = []
        for i in range(1000): # NOTE: all*
            if i // 5 % 20 in self._EVAL_SKIP_OBJECT: # NOTE: all - removing ungraspable objects by gripper
                continue
            if i // 100 in subject_ind and i % 100 in sequence_ind:
                if mano_side == ["right", "left"]:
                    self.scene_ids.append(i)
                else:
                    if i % 5 != 4:
                        if (
                            i % 5 in (0, 1)
                            and mano_side == ["right"]
                            or i % 5 in (2, 3)
                            and mano_side == ["left"]
                        ):
                            self.scene_ids.append(i)
                    elif mano_side == self.dex_ycb.load_meta_from_cache(i)["mano_sides"]:
                        self.scene_ids.append(i)

    @property
    def num_scenes(self):
        return len(self.scene_ids)

    @property
    def scene_ids(self):
        return self._scene_ids

    def reset(self, env_ids=None, **kwargs):
        if "idx" in kwargs:
            assert "scene_id" not in kwargs
            kwargs["scene_id"] = self.scene_ids[kwargs["idx"]]
            del kwargs["idx"]
        else:
            assert kwargs["scene_id"] in self.scene_ids

        return super().reset(env_ids=env_ids, **kwargs)


class BenchmarkRLBenchRecorder:

    def __init__(self, cfg):
        self._cfg = cfg

        self._env = HandoverBenchmarkWrapperFiltered(gym.make(self._cfg.ENV.ID, cfg=self._cfg))

    def _draw_gripper_position(self, pose):
        body = easysim.Body()
        body.name = "gripper"
        body.geometry_type = easysim.GeometryType.SPHERE
        body.sphere_radius = 0.1
        body.initial_base_position = pose
        body.link_color = [(0.19, 0.85, 0.21, 0.5)]
        body.link_collision_filter = [0]
        self._env.scene.add_body(body)
        self._gripper_body = body

    def _draw_object_position(self, pose):
        body = easysim.Body()
        body.name = "object"
        body.geometry_type = easysim.GeometryType.SPHERE
        body.sphere_radius = 0.1
        body.initial_base_position = pose
        body.link_color = [(0.19, 0.21, 0.85, 0.5)]
        body.link_collision_filter = [0]
        self._env.scene.add_body(body)
        self._object_body = body

    def run(self, policy, index=None, record_dir = None):
        self._render_steps = int((1.0 / self._cfg.BENCHMARK.RENDER_FRAME_RATE) / self._cfg.SIM.TIME_STEP)
        self._steps_wait = int(self._cfg.BENCHMARK.TIME_WAIT / self._cfg.SIM.TIME_STEP)

        if self._cfg.BENCHMARK.RECORD:
            if not record_dir:
                raise ValueError("'record_dir' must be specified through config BENCHMARK.RECORD_DIR")
            if self._cfg.SIM.RENDER:
                raise ValueError("RECORD can only be True when SIM.RENDER is set to False")
            if self._cfg.BENCHMARK.RECORD_DIR is None:
                raise ValueError("'record_dir' must be specified when RECORD is set to True or RECORD must be False")
            if 1.0 / self._cfg.BENCHMARK.RENDER_FRAME_RATE < self._cfg.SIM.TIME_STEP:
                raise ValueError("Offscreen record time step must not be smaller than TIME_STEP")
            if not os.path.exists(self._cfg.BENCHMARK.RECORD_DIR):
                raise ValueError(f"data_folder: '{self._cfg.BENCHMARK.RECORD_DIR}' does not exist.")
            record_dir = self._cfg.BENCHMARK.RECORD_DIR
            
        if index is None:
            indices = range(self._env.num_scenes)
        else:
            indices = [index]

        for idx in indices:
            print("{:04d}/{:04d}: scene {}".format(idx + 1, self._env.num_scenes, self._env.scene_ids[idx]))

            result, elapsed_time = self._run_scene(idx, policy, record_dir)

            print("time:   {:6.2f}".format(elapsed_time))
            print("frame:  {:5d}".format(result["elapsed_frame"]))
            print("task:   {}".format(result["task"]))

            self._post_run_scene(result)

    @timer
    def _run_scene(self, idx, policy, data_folder=None):
        scene_idx = self._env.scene_ids[idx]
        obs = self._env.reset(idx=idx)
        policy.reset()
        self._frame_count = 0

        if self._cfg.BENCHMARK.DRAW_GOAL:
            self._draw_gripper_position(policy._get_ee_pose(obs))
            self._draw_object_position(policy._get_object_pose(obs))

        task_episode_path, task = self._pre_run_scene(obs, scene_idx, data_folder)

        result = {}
        result["action"] = []
        result["elapsed_time"] = []
        result["task_episode_data_path"] = task_episode_path
        result["task"] = task
        result["keypoints"] = {"approach": [],
                               "grasp": None}

        while True:

            (action, policy_info), elapsed_time = self._run_policy(policy, obs)
            if not policy._approach is None:
                result["keypoints"]["approach"].append([self._frame_count, policy._approach])

            if "obs_time" in policy_info:
                elapsed_time -= policy_info["obs_time"]

            result["action"].append(action)
            result["elapsed_time"].append(elapsed_time)

            # Update the Handover environment
            obs, _, _, env_info = self._env.step(action) # NOTE: info: status of Benchmark runner

            # Flag when gripper has grasped object -  NOTE: Changed in addition to 'panda.gripper_open'
            if not policy._done_frame is None:
                self._gripper_open = 0.0
                if result["keypoints"]["grasp"] is None:
                    result["keypoints"]["grasp"] = self._frame_count
            else:
                self._gripper_open = 1.0
            
            self._inter_run_scene(obs, result["task_episode_data_path"])

            if env_info["status"] != 0:
                self._gripper_open = 0.0
                self._inter_run_scene(obs, result["task_episode_data_path"], force_run=True) # Note: Reached end state so do another save here (with gripper_open = 0.0)
                break

        result["action"] = np.array(result["action"])
        result["elapsed_time"] = np.array(result["elapsed_time"])
        result["elapsed_frame"] = self._env.frame
        result["result"] = env_info["status"]
        
        return result
    
    def _pre_run_scene(self, initial_obs, scene_idx, data_folder=None):
        """Create task data folder"""

        task_object = initial_obs["ycb_classes"][list(initial_obs["ycb_bodies"])[0]].lstrip('0123456789_')
        task = [f"handing over {task_object}", f"grab the {task_object}", "handing over object"]

        task_episode_path = None

        if self._cfg.BENCHMARK.RECORD:

            # Create folder directory
            task_episodes_path = os.path.join(data_folder,
                                            task[0].replace(" ", "_"),
                                            "all_variations",
                                            "episodes")
            task_episode_path = os.path.join(task_episodes_path, "episode%d" % scene_idx)

            self._check_and_mkdirs(task_episode_path)

            self.rlbench_observations = []

        return task_episode_path, task
    
    def _inter_run_scene(self, obs, task_episode_data_path, force_run=False):

        if force_run or (self._env.frame >= self._steps_wait and (self._env.frame % self._render_steps) <= ((self._env.frame - 1) % self._render_steps)):

            if self._cfg.SIM.RENDER and not self._cfg.BENCHMARK.RECORD:
                scene_camera = obs["callback_render_camera_scene"](0)
            if self._cfg.BENCHMARK.RECORD:
                self._save_render(task_episode_data_path, obs)
                rlbench_observation = self._save_rlbench_observation(obs)
                self.rlbench_observations.append(rlbench_observation)

            self._frame_count += 1
    
    def _post_run_scene(self, result):
        
        if result["result"] == EpisodeStatus.SUCCESS:
            print("result:  success")

            if self._cfg.BENCHMARK.RECORD:
                variation_number = 0
                rlbench_demo = Demo(self.rlbench_observations)
                rlbench_demo.variation_number = variation_number
                rlbench_demo.keypoints = result["keypoints"]# NOTE: ADD keypoints manually

                low_dim_obs_path = os.path.join(result["task_episode_data_path"], 'low_dim_obs.pkl')
                with open(low_dim_obs_path, 'wb') as f:
                    pickle.dump(rlbench_demo, f)

                variation_number_path = os.path.join(result["task_episode_data_path"], 'variation_number.pkl')
                with open(variation_number_path, 'wb') as f:
                    pickle.dump(variation_number, f)

                descriptions_path = os.path.join(result["task_episode_data_path"], 'variation_descriptions.pkl')
                with open(descriptions_path, 'wb') as f:
                    pickle.dump(result["task"], f)

        else:
            if result["result"] == EpisodeStatus.FAILURE_HUMAN_CONTACT:
                failure = "HUMAN_CONTACT"
            elif result["result"] == EpisodeStatus.FAILURE_OBJECT_DROP:
                failure = "OBJECT_DROP"
            elif result["result"] == EpisodeStatus.FAILURE_TIMEOUT:
                failure = "TIMEOUT"
            print("result:  failure {:s}".format(failure))
            
            if self._cfg.BENCHMARK.RECORD: # Remove task data folder if failed
                shutil.rmtree(result["task_episode_data_path"])
    
    def _save_render(self, render_dir, obs):

        cameras = ['wrist']
        for camera_i in range(self._cfg.ENV.PERAC_TOTAL_CAMERA_SCENES):
            cameras.append(camera_i)
        render_types = ['rgb', 'depth', 'mask']

        for camera in cameras:
            if camera == "wrist":
                if self._cfg.ENV.PERACT_RENDERER_CAMERA_WRIST_USE:
                    camera_renders = obs[f"callback_render_camera_{camera}"]()
                # camera_near = obs[f"callback_camera_{camera}_near"]()
                # camera_far = obs[f"callback_camera_{camera}_far"]()
                    camera = f"{camera}"
                else:
                    continue # Skip wrist camera
            else:
                camera_renders = obs[f"callback_render_camera_scene"](camera)
                # camera_near = obs[f"callback_camera_scene_near"]()
                # camera_far = obs[f"callback_camera_scene_far"]()
                camera = f"view_{camera}"

            for render_type, camera_render in zip(render_types, camera_renders):
                render_dir_camera = os.path.join(render_dir, f"{camera}_{render_type}")
                self._check_and_mkdirs(render_dir_camera)

                render_file = os.path.join(render_dir_camera, "{:d}.png".format(self._frame_count))

                if render_type == "rgb":
                    camera_render = camera_render[:,:,::-1]
                    # cv2.imwrite(render_file, camera_render)
                elif render_type == "depth":
                    camera_render = camera_render * 1000 #np.clip(camera_render, 0, None) * 1000
                    camera_render = camera_render.astype(np.uint16)
                elif render_type == "mask":
                    continue # Skip mask

                cv2.imwrite(render_file, camera_render)

    def _save_rlbench_observation(self, obs):

        misc = dict()
        
        if self._cfg.ENV.PERACT_RENDERER_CAMERA_WRIST_USE:
            misc['wrist_camera_intrinsics'] = obs["callback_camera_wrist_intrinsics"]()
            misc['wrist_camera_extrinsics'] = obs["callback_camera_wrist_extrinsics"]()
            misc['wrist_camera_near'] = obs["callback_camera_wrist_near"]()
            misc['wrist_camera_far'] = obs["callback_camera_wrist_far"]()

        misc['object_state'] = obs["ycb_bodies"][list(obs["ycb_bodies"])[0]].link_state[0,-1,0:7].numpy()

        for camera_i in range(self._cfg.ENV.PERAC_TOTAL_CAMERA_SCENES):
            view_i = f"view_{camera_i}"
            misc[f'{view_i}_camera_intrinsics'] = obs["callback_camera_scene_intrinsics"](camera_i)
            misc[f'{view_i}_camera_extrinsics'] = obs["callback_camera_scene_extrinsics"](camera_i)
            misc[f'{view_i}_camera_near'] = 0 #obs["callback_camera_scene_near"](camera_i)
            misc[f'{view_i}_camera_far'] = 1 #obs["callback_camera_scene_far"](camera_i)

        observation = Observation(left_shoulder_rgb=None,
                                  left_shoulder_depth=None,
                                  left_shoulder_mask=None,
                                  left_shoulder_point_cloud=None,
                                  right_shoulder_rgb=None,
                                  right_shoulder_depth=None,
                                  right_shoulder_mask=None,
                                  right_shoulder_point_cloud=None,
                                  overhead_rgb=None,
                                  overhead_depth=None,
                                  overhead_mask=None,
                                  overhead_point_cloud=None,
                                  wrist_rgb=None,
                                  wrist_depth=None,
                                  wrist_mask=None,
                                  wrist_point_cloud=None,
                                  front_rgb=None,
                                  front_depth=None,
                                  front_mask=None,
                                  front_point_cloud=None,
                                  joint_velocities=obs["callback_joint_velocities"](),
                                  joint_positions=obs["callback_joint_velocities"](),
                                  joint_forces=None,
                                  gripper_open=self._gripper_open, #obs["callback_gripper_open"](),
                                  gripper_pose=obs["callback_gripper_pose"](),
                                  gripper_matrix=self._pose_to_4x4mat(obs["callback_gripper_pose"]()),
                                  gripper_joint_positions=obs["callback_finger_positions"](),
                                  gripper_touch_forces=None,
                                  task_low_dim_state=None,
                                  ignore_collisions=True, # TODO: fix
                                  misc=misc)

        return observation
    
    def _pose_to_4x4mat(self, pose):
        transformation_matrix = np.eye(4)

        try:
            transformation_matrix[:3, 3] = pose[:3]
            rotation = Rot.from_quat(pose[3:])  # Quaternion format (x, y, z, w)
            rotation_matrix_3x3 = rotation.as_matrix()  # Get as 3x3 matrix
            transformation_matrix[:3, :3] = rotation_matrix_3x3

        except Exception as error:
            print(error)
        
        return transformation_matrix

    def _check_and_mkdirs(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    @timer
    def _run_policy(self, policy, obs):
        return policy.forward(obs)