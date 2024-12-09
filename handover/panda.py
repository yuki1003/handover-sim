# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""

Derived from:
https://github.com/bryandlee/franka_pybullet/blob/7c66ad1a211a4c118bc58eb374063f7f55f60973/src/panda_gripper.py
https://github.com/liruiw/OMG-Planner/blob/dcbbb8279570cd62cf7388bf393c8b3e2d5686e5/bullet/panda_gripper.py
"""

import easysim
import os
import numpy as np
import torch

from scipy.spatial.transform import Rotation as Rot

from handover.transform3d import get_t3d_from_qt


class Panda:
    _URDF_FILE = os.path.join(
        os.path.dirname(__file__), "data", "assets", "franka_panda", "panda_gripper.urdf"
    )
    _RIGID_SHAPE_COUNT = 11

    LINK_IND_HAND = 8
    LINK_IND_FINGERS = (9, 10)

    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene

        body = easysim.Body()
        body.name = "panda"
        body.geometry_type = easysim.GeometryType.URDF
        body.urdf_file = self._URDF_FILE
        body.use_fixed_base = True
        body.use_self_collision = True
        body.initial_base_position = (
            self._cfg.ENV.PANDA_BASE_POSITION + self._cfg.ENV.PANDA_BASE_ORIENTATION
        )
        body.initial_dof_position = self._cfg.ENV.PANDA_INITIAL_POSITION
        body.initial_dof_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) # NOTE: Last 2 are grippers
        body.link_collision_filter = [
            self._cfg.ENV.COLLISION_FILTER_PANDA
        ] * self._RIGID_SHAPE_COUNT
        body.dof_control_mode = easysim.DoFControlMode.POSITION_CONTROL
        body.dof_position_gain = self._cfg.ENV.PANDA_POSITION_GAIN
        body.dof_velocity_gain = self._cfg.ENV.PANDA_VELOCITY_GAIN
        body.dof_max_force = self._cfg.ENV.PANDA_MAX_FORCE
        self._scene.add_body(body)
        self._body = body

    @property
    def body(self):
        return self._body

    def step(self, dof_target_position):
        self.body.dof_target_position = dof_target_position


class PandaHandCamera(Panda):
    _URDF_FILE = os.path.join(
        os.path.dirname(__file__),
        "data",
        "assets",
        "franka_panda",
        "panda_gripper_hand_camera.urdf",
    )
    _RIGID_SHAPE_COUNT = 12

    LINK_IND_CAMERA = 11

    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)

        camera = easysim.Camera()
        camera.name = "panda_hand_camera"
        camera.width = 224
        camera.height = 224
        camera.vertical_fov = 90
        camera.near = 0.035
        camera.far = 2.0
        camera.position = [(0.0, 0.0, 0.0)]
        camera.orientation = [(0.0, 0.0, 0.0, 1.0)]
        self._scene.add_camera(camera)
        self._camera = camera

        # Get rotation from URDF to OpenGL view frame.
        orn = Rot.from_euler("XYZ", (-np.pi / 2, 0.0, -np.pi)).as_quat().astype(np.float32)
        self._quat_urdf_to_opengl = torch.from_numpy(orn)

        # Get deproject points before depth multiplication.
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self._camera.height / 2 / np.tan(np.deg2rad(self._camera.vertical_fov) / 2)
        K[1, 1] = self._camera.height / 2 / np.tan(np.deg2rad(self._camera.vertical_fov) / 2)
        K[0, 2] = self._camera.width / 2
        K[1, 2] = self._camera.height / 2
        K_inv = np.linalg.inv(K)
        x, y = np.meshgrid(np.arange(self._camera.width), np.arange(self._camera.height))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        ones = np.ones((self._camera.height, self._camera.width), dtype=np.float32)
        xy1s = np.stack((x, y, ones), axis=2).reshape(self._camera.width * self._camera.height, 3).T
        self._deproject_p = np.matmul(K_inv, xy1s).T

        # Get transform from hand to pinhole camera frame.
        pos = (+0.036, 0.0, +0.036)
        orn = Rot.from_euler("XYZ", (0.0, 0.0, +np.pi / 2)).as_quat().astype(np.float32)
        self._t3d_hand_to_camera = get_t3d_from_qt(orn, pos)

    def get_point_states(self, segmentation_ids):
        # Get OpenGL view frame from URDF camera frame.
        pos = self.body.link_state[0, self.LINK_IND_CAMERA, 0:3]
        orn = self.body.link_state[0, self.LINK_IND_CAMERA, 3:7]
        orn = _quaternion_multiplication(orn, self._quat_urdf_to_opengl)

        # Set camera pose.
        self._camera.update_attr_array("position", torch.tensor([0]), pos)
        self._camera.update_attr_array("orientation", torch.tensor([0]), orn)

        # Render camera image.
        depth = self._camera.depth[0].numpy()
        segmentation = self._camera.segmentation[0].numpy()

        point_states = []

        for segmentation_id in segmentation_ids:
            # Get point state in pinhole camera frame.
            mask = segmentation == segmentation_id
            point_state = (
                np.tile(depth[mask].reshape(-1, 1), (1, 3)) * self._deproject_p[mask.ravel(), :]
            )

            # Transform point state to hand frame.
            point_state = self._t3d_hand_to_camera.transform_points(point_state)

            point_states.append(point_state)

        return point_states
    

class PandaPerActCamera(Panda):
    _URDF_FILE = os.path.join(
        os.path.dirname(__file__),
        "data",
        "assets",
        "franka_panda",
        "panda_gripper_hand_camera.urdf",
    )
    _RIGID_SHAPE_COUNT = 12

    LINK_IND_CAMERA = 11

    _DEPTH_SCALE = 1000 # Convert depth scale meters into millimeters

    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)

        self.fingers_default_distance = 0.08

        self.setup_wrist_camera()

        self._cameras = []
        self.setup_scene_cameras(cfg)

        # Get rotation from URDF to OpenGL view frame.
        orn = Rot.from_euler("XYZ", (-np.pi / 2, 0.0, -np.pi)).as_quat().astype(np.float32)
        self._quat_urdf_to_opengl = torch.from_numpy(orn)

    def setup_wrist_camera(self):

        camera_wrist = PerActCamera()
        camera_wrist.name = "wrist_cam"
        camera_wrist.width = 128
        camera_wrist.height = 128
        camera_wrist.vertical_fov = 90
        camera_wrist.near = 0.035 # Minimum required distance to not clip with camera/panda
        camera_wrist.far = 2.0
        camera_wrist.position = [(0.0, 0.0, 0.0)] # NOTE: Overwritten later
        camera_wrist.orientation = [(0.0, 0.0, 0.0, 1.0)] # NOTE: Overwritten later
        self._scene.add_camera(camera_wrist)
        self._camera_wrist = camera_wrist

    def setup_scene_cameras(self, cfg):

        camera_width = cfg.ENV.RENDERER_CAMERA_WIDTH
        camera_height = cfg.ENV.RENDERER_CAMERA_HEIGHT
        camera_vertical_fov = cfg.ENV.RENDERER_CAMERA_VERTICAL_FOV
        camera_near = cfg.ENV.RENDERER_CAMERA_NEAR
        camera_far = cfg.ENV.RENDERER_CAMERA_FAR
        camera_up_vector = (0.0, 0.0, 1.0)
        camera_target = list(cfg.BENCHMARK.GOAL_CENTER)
        camera_target[1] += 0.2

        radius = cfg.ENV.PERACT_RENDERER_CAMERA_SCENE_DISTANCE_HOR
        height = cfg.ENV.PERACT_RENDERER_CAMERA_SCENE_DISTANCE_VER
        angles = np.linspace(0, 2 * np.pi, cfg.ENV.PERACT_RENDERER_CAMERA_SCENE_AMOUNT, endpoint=False)

        for camera_i, angle in enumerate(angles):
            camera = PerActCamera()
            camera.width = camera_width
            camera.height = camera_height
            camera.vertical_fov = camera_vertical_fov
            camera.near = camera_near
            camera.far = camera_far
            camera.up_vector = camera_up_vector
            
            camera.name = f"panda_camera_side_{camera_i}"
            camera.position = (camera_target[0] + radius*np.cos(angle),
                               camera_target[1] + radius*np.sin(angle),
                               camera_target[2] + height)
            camera.target = camera_target

            self._scene.add_camera(camera)
            self._cameras.append(camera)

    def render_camera_wrist(self):
        # Get OpenGL view frame from URDF camera frame.
        pos = self.body.link_state[0, self.LINK_IND_CAMERA, 0:3]
        orn = self.body.link_state[0, self.LINK_IND_CAMERA, 3:7]
        orn = _quaternion_multiplication(orn, self._quat_urdf_to_opengl)

        # Set camera pose.
        self._camera_wrist.update_attr_array("position", torch.tensor([0]), pos)
        self._camera_wrist.update_attr_array("orientation", torch.tensor([0]), orn)

        # Render camera image.
        color = self._camera_wrist.color[0].numpy()[:,:,:3]
        depth = self._camera_wrist.depth[0].numpy()
        # depth = np.ones_like(self._camera_wrist.depth[0].numpy()) * self._camera_wrist.near
        segmentation = self._camera_wrist.segmentation[0].numpy()

        return (color, depth, segmentation)
    
    def camera_wrist_intrinsics(self):
        return self._camera_wrist.intrinsic_matrix

    def camera_wrist_extrinsics(self):
        # Get OpenGL view frame from URDF camera frame.
        pos = self.body.link_state[0, self.LINK_IND_CAMERA, 0:3]
        orn = self.body.link_state[0, self.LINK_IND_CAMERA, 3:7]
        orn = _quaternion_multiplication(orn, self._quat_urdf_to_opengl)

        # Set camera pose.
        self._camera_wrist.update_attr_array("position", torch.tensor([0]), pos)
        self._camera_wrist.update_attr_array("orientation", torch.tensor([0]), orn)
        
        return self._camera_wrist.extrinsic_matrix
    
    def camera_wrist_near(self):
        return self._camera_wrist.near
    
    def camera_wrist_far(self):
        return self._camera_wrist.far
    

    def render_camera_scene(self, camera_number):
        
        # Render camera image.
        color = self._cameras[camera_number].color[0].numpy()[:,:,:3]
        depth = self._cameras[camera_number].depth[0].numpy()
        segmentation = self._cameras[camera_number].segmentation[0].numpy()

        return (color, depth, segmentation)
    
    def camera_scene_intrinsics(self, camera_number):
        return self._cameras[camera_number].intrinsic_matrix
    
    def camera_scene_extrinsics(self, camera_number):
        return self._cameras[camera_number].extrinsic_matrix
    
    def camera_scene_near(self, camera_number):
        return 0#self._cameras[camera_number].near
    
    def camera_scene_far(self, camera_number):
        return 1#self._cameras[camera_number].far
    
    
    def gripper_pose(self):
        return self.body.link_state[0, self.LINK_IND_HAND, 0:7].numpy()
    
    def gripper_open(self):
        """Return gripper open. 1.0 if open, else 0.0"""
        # NOTE: This is either dynamic or changes state @ beginnin. We want it when the gripper is actually closed
        left_finger_pos = self.body.link_state[0,self.LINK_IND_FINGERS[0], 0:3].numpy()
        right_finger_pos = self.body.link_state[0,self.LINK_IND_FINGERS[1], 0:3].numpy()
        gripper_dist_norm = np.linalg.norm(right_finger_pos - left_finger_pos) / self.fingers_default_distance
        # gripper_open = round(gripper_dist_norm,2)
        gripper_open = float(gripper_dist_norm > 0.7) # NOTE: May be different per object
        return gripper_open

    def dof_force(self): #TODO: CHECK IF NECESSARY
        return self.body.dof_force
    
    def joint_angles(self):
        return self.body.dof_state[0, :-2, 0].numpy()
    
    def joint_velocities(self):
        return self.body.dof_state[0, :-2, 1].numpy()
    
    def finger_positions(self):
        return self.body.dof_state[0, -2:, 0].numpy()
    
class PerActCamera(easysim.Camera):

    @property
    def intrinsic_matrix(self):
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self._height / 2 / np.tan(np.deg2rad(self._vertical_fov) / 2)
        K[1, 1] = self._height / 2 / np.tan(np.deg2rad(self._vertical_fov) / 2)
        K[0, 2] = self._width / 2
        K[1, 2] = self._height / 2
        return K
    
    @property
    def extrinsic_matrix(self):
        if self._target is not None or self._up_vector is not None:
            position, target, up = np.array(self._position), np.array(self._target), np.array(self.up_vector)
            
            # return self.compute_extrinsic_matrix(position, target)
            return self._position_target_up_to_4x4mat(position, target, up)
            
        elif self._orientation is not None:
            position_x, position_y, position_z = np.array(self._position[0])
            x, y, z, w = self._orientation[0]
            camera_pose = np.array([position_x,position_y,position_z,x,y,z,w])

            return self._pose_to_4x4mat(camera_pose)

        else:
            raise NotImplementedError
    
    def _pose_to_4x4mat(self,pose):
        transformation_matrix = np.eye(4)

        transformation_matrix[:3, 3] = pose[:3]
        rotation = Rot.from_quat(pose[3:])  # Quaternion format (x, y, z, w)
        rotation_matrix_3x3 = rotation.as_matrix()  # Get as 3x3 matrix
        transformation_matrix[:3, :3] = rotation_matrix_3x3
        
        return transformation_matrix
    
    def _position_target_up_to_4x4mat(self,position, target, up):
        transformation_matrix = np.eye(4)

        forward = np.array(target) - np.array(position)
        forward /= np.linalg.norm(forward)
        
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        
        transformation_matrix[:3, 0] = -right
        transformation_matrix[:3, 1] = up_corrected
        transformation_matrix[:3, 2] = forward
        transformation_matrix[:3, 3] = position

        return transformation_matrix


def _quaternion_multiplication(q1, q2):
    q1x, q1y, q1z, q1w = torch.unbind(q1, axis=-1)
    q2x, q2y, q2z, q2w = torch.unbind(q2, axis=-1)
    return torch.stack(
        (
            q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
            q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x,
            q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w,
            q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z,
        ),
        axis=-1,
    )
