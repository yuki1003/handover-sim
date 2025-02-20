# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import os
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'

sys.path = [p for p in sys.path if '/peract/' not in p]
sys.path.append("/home/begroup/Projects/PerAct_ws/peract_colab")

from matplotlib import pyplot as plt

import gym
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import torch

from agent.peract_agent import PerceiverActorAgent
from agent.perceiver_io import PerceiverIO

from arm.utils import visualise_voxel, get_gripper_render_pose
from arm.augmentation import matrix_to_quaternion, quaternion_to_matrix

import handover
from handover.config_peract import get_config_from_args
from handover.benchmark_runner import BenchmarkRunner, timer
from handover.benchmark_recorder import HandoverBenchmarkWrapperFiltered

import pybullet
import pybullet_utils.bullet_client as bullet_client

from demo_benchmark_wrapper import start_conf, time_wait
# from examples.demo_benchmark_wrapper import start_conf, time_wait

# Use urdf file for inverse kinematics
panda_urdf_file = os.path.join(
    os.path.dirname(handover.__file__), "data", "assets", "franka_panda", "panda_gripper.urdf"
)


def quat_loss(q1, q2):
    return 1 - np.power(np.sum(q1 * q2, axis=-1), 2)


class ApproachRegionCondition:
    def __init__(self, slope=10.0, pos_tol=1.5e-2, max_pos_tol=5e-2, theta_tol=np.radians(10.0)):
        self._slope = slope
        self._pos_tol = pos_tol
        self._max_pos_tol = max_pos_tol
        self._theta_tol = theta_tol

        self._start_pose = None
        self._final_pose = None

    def set_region(self, pose0, pose1):
        self._start_pose = pose0
        self._final_pose = pose1

    def __call__(self, pose):
        if self._start_pose is None or self._final_pose is None:
            return False

        actor_pos = pose[0:3]
        actor_rot = pose[3:7]

        frame_pos = self._start_pose[0:3]
        frame_rot = self._start_pose[3:7]
        grasp_pos = self._final_pose[0:3]
        grasp_rot = self._final_pose[3:7]

        theta_frame = quat_to_angle(actor_rot, frame_rot)
        theta_grasp = quat_to_angle(actor_rot, grasp_rot)
        theta = min(abs(theta_frame), abs(theta_grasp))

        a = actor_pos - frame_pos
        b = grasp_pos - frame_pos
        l2 = np.linalg.norm(grasp_pos - frame_pos, ord=2)
        proj = np.dot(a, b) / l2
        dist_line = max(min(proj, 1.0), 0.0) / l2
        proj_pt = frame_pos + dist_line * (grasp_pos - frame_pos)
        dist = np.linalg.norm(proj_pt - actor_pos)
        pos_tol = min(self._pos_tol + (self._pos_tol * self._slope * l2), self._max_pos_tol)
        ok = dist < pos_tol and theta < self._theta_tol

        return ok


def quat_to_angle(q1, q2):
    res = 2 * (np.sum(q1 * q2)) ** 2 - 1
    res = np.clip(res, -1.0, +1.0)
    return np.arccos(res)


class AtPoseCondition:
    def __init__(self, position_tol, rotation_tol):
        self._position_tol = position_tol
        self._rotation_tol = rotation_tol

        self._goal_pose = None

    def set_goal(self, goal_pose):
        self._goal_pose = goal_pose

    def __call__(self, pose):
        actor_pos = pose[0:3]
        actor_rot = pose[3:7]

        grasp_pos = self._goal_pose[0:3]
        grasp_rot = self._goal_pose[3:7]

        dist = np.linalg.norm(grasp_pos - actor_pos)
        theta_grasp = quat_to_angle(actor_rot, grasp_rot)

        return dist < self._position_tol and theta_grasp < self._rotation_tol


class BulletPanda:
    def __init__(self, urdf_file, base_pos, base_orn):
        self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._id = self._p.loadURDF(urdf_file)
        self._p.resetBasePositionAndOrientation(self._id, base_pos, base_orn)

        self._ee_idx = 7

    def ik(self, q0, pos, rot=None, tol=1e-3, theta_tol=0.1, max_iter=1000):
        self._set_joint_position(q0)
        for _ in range(max_iter):
            kwargs = {"restPoses": q0}
            if rot is not None:
                kwargs["targetOrientation"] = rot
            conf = self._p.calculateInverseKinematics(self._id, self._ee_idx, pos, **kwargs)
            pos_fk, rot_fk = self._fk(conf)
            dist = np.linalg.norm(pos_fk - pos)
            if dist < tol and (rot is None or quat_to_angle(rot_fk, rot) < theta_tol):
                return conf[: self._ee_idx]
            q0 = conf
        return None

    def _set_joint_position(self, position):
        for i in range(self._ee_idx):
            self._p.resetJointState(self._id, i, position[i])

    def _fk(self, q):
        self._set_joint_position(q)
        state = self._p.getLinkState(self._id, self._ee_idx)
        pos, rot = state[:2]
        return np.array(pos), np.array(rot)


def compose_qq(q1, q2):
    qww = q1[..., 6] * q2[..., 6]
    qxx = q1[..., 3] * q2[..., 3]
    qyy = q1[..., 4] * q2[..., 4]
    qzz = q1[..., 5] * q2[..., 5]

    q1w2x = q1[..., 6] * q2[..., 3]
    q2w1x = q2[..., 6] * q1[..., 3]
    q1y2z = q1[..., 4] * q2[..., 5]
    q2y1z = q2[..., 4] * q1[..., 5]

    q1w2y = q1[..., 6] * q2[..., 4]
    q2w1y = q2[..., 6] * q1[..., 4]
    q1z2x = q1[..., 5] * q2[..., 3]
    q2z1x = q2[..., 5] * q1[..., 3]

    q1w2z = q1[..., 6] * q2[..., 5]
    q2w1z = q2[..., 6] * q1[..., 5]
    q1x2y = q1[..., 3] * q2[..., 4]
    q2x1y = q2[..., 3] * q1[..., 4]

    q3 = np.zeros(np.broadcast_shapes(q1.shape, q2.shape))
    q3[..., 0:3] = compose_qp(q1, q2[..., 0:3])
    q3[..., 3] = q1w2x + q2w1x + q1y2z - q2y1z
    q3[..., 4] = q1w2y + q2w1y + q1z2x - q2z1x
    q3[..., 5] = q1w2z + q2w1z + q1x2y - q2x1y
    q3[..., 6] = qww - qxx - qyy - qzz

    return q3


def compose_qp(q, pt):
    """
    Apply a 3D transformation (translation and rotation) to a set of points.

    This function takes a 3D transformation represented by a translation vector
    and a quaternion (contained in `q`) and applies it to a set of 3D points (`pt`).
    
    Parameters:
        q (ndarray): Array of shape (..., 7) representing the transformations.
                     - The first 3 components are the translation vector (x, y, z).
                     - The last 4 components are the quaternion (qx, qy, qz, qw) defining rotation.
        pt (ndarray): Array of shape (..., 3) representing the 3D points to transform.
                      Each point has coordinates (px, py, pz).
    
    Returns:
        ndarray: Transformed points with the same shape as `pt`, adjusted by the
                 translation and rotation described by `q`.
    """
    px = pt[..., 0]
    py = pt[..., 1]
    pz = pt[..., 2]

    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    qx = q[..., 3]
    qy = q[..., 4]
    qz = q[..., 5]
    qw = q[..., 6]

    qxx = qx**2
    qyy = qy**2
    qzz = qz**2
    qwx = qw * qx
    qwy = qw * qy
    qwz = qw * qz
    qxy = qx * qy
    qxz = qx * qz
    qyz = qy * qz

    pt2 = np.zeros((*np.broadcast_shapes(q.shape[:-1], pt.shape[:-1]), 3))
    pt2[..., 0] = x + px + 2 * ((-1 * (qyy + qzz) * px) + ((qxy - qwz) * py) + ((qwy + qxz) * pz))
    pt2[..., 1] = y + py + 2 * (((qwz + qxy) * px) + (-1 * (qxx + qzz) * py) + ((qyz - qwx) * pz))
    pt2[..., 2] = z + pz + 2 * (((qxz - qwy) * px) + ((qwx + qyz) * py) + (-1 * (qxx + qyy) * pz))

    return pt2


def simple_extend(q1, q2, step_size=0.1):
    """
    Incrementally move from one point toward another with a specified step size.

    This function computes an intermediate point `q3` along the straight-line 
    path from `q1` to `q2`, such that the step size is limited to `step_size`. 
    If the distance between `q1` and `q2` is less than `step_size`, the function 
    directly returns `q2`.

    Parameters:
        q1 (ndarray): Starting point in n-dimensional space, as a NumPy array.
        q2 (ndarray): Target point in n-dimensional space, as a NumPy array.
        step_size (float, optional): Maximum distance to move from `q1` toward `q2`.
                                     Defaults to 0.1.

    Returns:
        ndarray: The new point `q3` that lies on the straight-line path from `q1`
                 to `q2`, no farther than `step_size` from `q1`.
    """
    dq = q2 - q1
    dist = np.linalg.norm(dq)
    if dist < step_size:
        return q2
    else:
        q3 = q1.copy()
        q3 += (dq / dist) * step_size
        return q3


class BenchmarkRunnerPerAct(BenchmarkRunner):

    def __init__(self, cfg):
        self._cfg = cfg

        self._env = HandoverBenchmarkWrapperFiltered(gym.make(self._cfg.ENV.ID, cfg=self._cfg))
    
    @timer
    def _run_scene(self, idx, policy, render_dir=None):
        obs = self._env.reset(idx=idx)
        policy.reset()

        result = {}
        result["action"] = []
        result["elapsed_time"] = []
        frame_count = 0

        if self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER:
            self._render_offscreen_and_save(render_dir)

        while True:

            obs["frame_count"] = frame_count
            (action, info), elapsed_time = self._run_policy(policy, obs)
            frame_count += 1

            if "obs_time" in info:
                elapsed_time -= info["obs_time"]

            result["action"].append(action)
            result["elapsed_time"].append(elapsed_time)

            obs, _, _, info = self._env.step(action)

            if (
                self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER
                and (self._env.frame % self._render_steps)
                <= (self._env.frame - 1) % self._render_steps
            ):
                self._render_offscreen_and_save(render_dir)

            if info["status"] != 0:
                break

        result["action"] = np.array(result["action"])
        result["elapsed_time"] = np.array(result["elapsed_time"])
        result["elapsed_frame"] = self._env.frame
        result["result"] = info["status"]

        return result


class PerActAgent:
    def __init__(self, cfg, time_close_gripper=0.5, device="cpu"):
        self._cfg = cfg
        self._steps_wait = int(self._cfg.BENCHMARK.TIME_WAIT / self._cfg.SIM.TIME_STEP)
        self._steps_action_repeat = int(self._cfg.BENCHMARK.TIME_ACTION_REPEAT / self._cfg.SIM.TIME_STEP)
        self._steps_close_gripper = int(time_close_gripper / self._cfg.SIM.TIME_STEP)

        self._start_position = self._cfg.ENV.PANDA_INITIAL_POSITION

        self._in_approach_region = ApproachRegionCondition(
            slope=50.0, pos_tol=1.5e-2, max_pos_tol=4e-2, theta_tol=np.radians(15.0)
        )
        self._at_grasp_pose = AtPoseCondition(position_tol=0.005, rotation_tol=np.radians(15.0))

        self._bullet_panda = BulletPanda(
            panda_urdf_file, self._cfg.ENV.PANDA_BASE_POSITION, self._cfg.ENV.PANDA_BASE_ORIENTATION
        )

        self._perceiver_encoder = PerceiverIO(**self._cfg.AGENT.PERCEIVOR_IO)
        
        self._agent = PerceiverActorAgent(perceiver_encoder=self._perceiver_encoder, **self._cfg.AGENT.PERACT)
        self._agent.build(training=False, device=device, language_goal="handing over banana")
        self._agent.load_weights(self._cfg.AGENT.model_path)


    @property
    def name(self):
        return "PerceiverActorAgent"
    
    def reset(self):
        self._done = False
        self._done_frame = None
        self._action_repeat = None
        self._back = None
        self._predicted_ee_pose = None

        self._count = 1

    def forward(self, obs):
        
        # Wait
        if obs["frame"] < self._steps_wait:
            action = self._start_position
        else:
            # Approach
            if not self._done:
                if (obs["frame"] - self._steps_wait) % self._steps_action_repeat == 0:
                    current_cfg = self._get_current_cfg(obs)
                    ee_pose = self._get_ee_pose(obs)
                    # gripper_joint_positions = current_cfg[7:9]
                    # gripper_open = True if sum(gripper_joint_positions) > 0.8 else False
                    # timestep = int(obs["frame_count"]/100)
                    timestep = self._count#int(obs["frame_count"]/100)
                    # self._count += 2
                    policy_obs = self._preprocess_obs(obs)
                    policy_obs["gripper_joint_positions"] = current_cfg[7:9]
                    policy_obs["gripper_open"] = True if sum(current_cfg[7:9]) > 0.8 else False
                    # object_pose = self._get_object_pose(obs)
                    action = self._get_policy_action(policy_obs, timestep, current_cfg, ee_pose)
                    self._action_repeat = action.copy()
                else:
                    action = self._action_repeat.copy()

            # Grasp + Back
            if self._done:
                if self._done_frame is None: # Capture time when at Grasp pose
                    self._done_frame = obs["frame"]

                if obs["frame"] < self._done_frame + self._steps_close_gripper: # Give time to close the gripper
                    current_cfg = self._get_current_cfg(obs)
                    action = current_cfg.copy()
                    action[7:9] = 0.0
                else: # Go Back
                    # if self._back is None:
                    current_cfg = self._get_current_cfg(obs)
                    self._back = self._get_back_slowly(current_cfg)
                    action = self._back.copy()

        return action, {}
    
    def _preprocess_obs(self, input_obs):
        
        obs = dict()
        
        render_types = ['rgb', 'depth', 'mask']

        for camera_i, camera_name in enumerate(self._cfg.AGENT.PERACT.camera_names):
            obs[f"{camera_name}_camera_intrinsics"] = input_obs["callback_camera_scene_intrinsics"](camera_i)
            obs[f"{camera_name}_camera_extrinsics"] = input_obs["callback_camera_scene_extrinsics"](camera_i)
            camera_renders = input_obs[f"callback_render_camera_scene"](camera_i)
            for render_type, camera_render in zip(render_types, camera_renders):
                if render_type == "rgb":
                    camera_render = camera_render.transpose(2, 0, 1)
                if render_type == "depth":
                    camera_render = np.expand_dims(camera_render, axis=0)
                obs[f"{camera_name}_{render_type}"] = camera_render

        return obs

    def _get_current_cfg(self, obs):
        return obs["panda_body"].dof_state[0, :, 0].numpy()

    def _get_object_pose(self, obs):
        return obs["ycb_bodies"][list(obs["ycb_bodies"])[0]].link_state[0, 6, 0:7].numpy()

    def _get_ee_pose(self, obs):
        return obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:7].numpy()

    def _compute_ik(self, pose, cfg):
        pos = pose[0:3]
        rot = pose[3:7]
        return self._bullet_panda.ik(cfg, pos, rot=rot)
    
    def _cfg_to_ee(self, cfg):
        ee_pos, ee_rot = self._bullet_panda._fk(cfg)
        return ee_pos, ee_rot
    
    def _get_policy_action(self, obs, timestep, current_cfg, ee_pose, debug=False):

        action = current_cfg.copy()
        action[7:9] = 0.04

        if not self._in_approach_region(ee_pose):

            print("How often is it in approach region?")
            (continuous_trans, continuous_quat, gripper_open, continuous_trans_confidence, continuous_quat_confidence), \
                (voxel_grid, coord_indices, rot_and_grip_indices, gripper_open) = self._agent.forward(obs, timestep)
            print("Prediction Confidence (trans):",continuous_trans_confidence)

            # Things to visualize NOTE DEBUG STUFF
            if debug:
                vis_voxel_grid = voxel_grid[0].detach().cpu().numpy()
                vis_trans_coord = coord_indices[0].detach().cpu().numpy().tolist()

                voxel_size = 0.045
                voxel_scale = voxel_size * 100
                gripper_pose_mat = get_gripper_render_pose(voxel_scale,
                                                        self._cfg.AGENT.PERACT.coordinate_bounds[:3],
                                                        continuous_trans,
                                                        continuous_quat)

                rendered_img_0 = visualise_voxel(vis_voxel_grid,
                                                None,
                                                [vis_trans_coord],
                                                None,
                                                voxel_size=voxel_size,
                                                rotation_amount=np.deg2rad(0),
                                                render_gripper=True,
                                                gripper_pose=gripper_pose_mat,
                                                gripper_mesh_scale=voxel_scale)

                rendered_img_270 = visualise_voxel(vis_voxel_grid,
                                                None,
                                                [vis_trans_coord],
                                                None,
                                                voxel_size=voxel_size,
                                                rotation_amount=np.deg2rad(45),
                                                render_gripper=True,
                                                gripper_pose=gripper_pose_mat,
                                                gripper_mesh_scale=voxel_scale)
                
                # Plot figures into a NumPy array
                fig = plt.figure(figsize=(20, 15))
                fig.add_subplot(1, 2, 1)
                plt.imshow(rendered_img_0)
                plt.axis('off')
                plt.title("Front view")
                fig.add_subplot(1, 2, 2)
                plt.imshow(rendered_img_270)
                plt.axis('off')
                plt.title("Side view")

                fig.savefig(f"timestep_{timestep}.png")
                plt.close()

            # predicted_ee_pose = np.concatenate((continuous_trans, continuous_quat))
            continuous_trans_grasp = self._move_ee_back(continuous_trans, continuous_quat, -0.05)
            self._predicted_ee_pose = np.concatenate((continuous_trans_grasp, continuous_quat)), self._to_ee_frame
            continuous_trans_approach = self._move_ee_back(continuous_trans, continuous_quat, 0.1)
            self._predicted_ee_pose_approach = np.concatenate((continuous_trans_approach, continuous_quat)), self._to_ee_frame

            self._in_approach_region.set_region(self._predicted_ee_pose_approach, self._predicted_ee_pose) # Set approach region
            self._at_grasp_pose.set_goal(self._predicted_ee_pose)
            
            approach_pose = self._predicted_ee_pose_approach.copy()
            approach_pose[0:3] = simple_extend(ee_pose[0:3], continuous_trans_approach[0:3], step_size = 0.1)
            ik_cfg = self._compute_ik(approach_pose, current_cfg)
            if ik_cfg is None:
                print(f"No feasible inverse kinematics found for {approach_pose}")
            action[0:7] = ik_cfg
        else:
            print("How often is it in grasp region?")
            # print(self._predicted_ee_pose_approach, self._predicted_ee_pose, ee_pose)
            if not self._at_grasp_pose(ee_pose):
                # Go to grasp pose.
                q_next = self._predicted_ee_pose.copy() # The prediction
                q_next[0:3] = simple_extend(ee_pose[0:3], self._predicted_ee_pose[0:3], step_size=0.05) # Move slowly forward
                ik_cfg = self._compute_ik(q_next, current_cfg) # Compute IK
                if ik_cfg is None:
                    print(f"No feasible inverse kinematics found for {q_next}")
                action[0:7] = ik_cfg # Update action
            else:
                # Move on to closing gipper and backing.
                self._done = True

        return action
    
    def _move_ee_back(self, continuous_trans, continuous_quat, distance):
        """move continuous_trans forward w.r.t. ee orientation by distance. Positive backward, negative forward"""
        pose_quat_wxyz = torch.from_numpy(np.concatenate((np.array([continuous_quat[3]]), continuous_quat[:3])))
        pose_rot_matrix = quaternion_to_matrix(pose_quat_wxyz).numpy()
        moved_continuous_trans = continuous_trans - distance * pose_rot_matrix[:,2]
        return moved_continuous_trans

    def _get_back(self, current_cfg):
        pos = self._cfg.BENCHMARK.GOAL_CENTER
        conf = self._bullet_panda.ik(current_cfg, pos)
        back = current_cfg.copy()
        back[0:7] = conf
        back[7:9] = 0.0
        return back
    
    def _get_back_slowly(self, current_cfg):
        current_ee_pos, _ = self._cfg_to_ee(current_cfg)
        pos = simple_extend(current_ee_pos, self._cfg.BENCHMARK.GOAL_CENTER, step_size=0.1)
        conf = self._bullet_panda.ik(current_cfg, pos)
        back = current_cfg.copy()
        back[0:7] = conf
        back[7:9] = 0.0
        return back


def main():
    cfg = get_config_from_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE: ",device)

    policy = PerActAgent(cfg, device=device) #TODO: Change

    benchmark_runner = BenchmarkRunnerPerAct(cfg)
    benchmark_runner.run(policy)


if __name__ == "__main__":
    main()
