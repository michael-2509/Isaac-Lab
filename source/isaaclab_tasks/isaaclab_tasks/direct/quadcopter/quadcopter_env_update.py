# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GPS-Denied Autonomous Navigation Environment for Quadcopter.
Uses hierarchical control with RL-based policy acting as a high-level planner.
The RL agent commands desired velocities (vx, vy, vz) and yaw rate, while a
low-level Geometric Controller handles attitude and thrust to achieve these targets.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_rotate, quat_from_matrix

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, SPHERE_MARKER_CFG  # isort: skip



class GeometricController:
    """Geometric tracking controller for quadcopter on SE(3).
    
    Converts desired velocity/yaw targets into thrust and body torques.
    Reference: Lee et al., "Geometric tracking control of a quadrotor UAV on SE(3)"
    """
    
    def __init__(self, cfg: QuadcopterEnvCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        # Controller gains (tuned for Crazyflie-like scale)
        # Position/Velocity gains - REDUCED to prevent thrust saturation
        self.kv = torch.tensor([0.8, 0.8, 1.2], device=device)  # FIX #1: Reduced from [2.0, 2.0, 1.0]
        self.kp = torch.tensor([5.0, 5.0, 5.0], device=device)  # Position error gain (unused in vel mode)
        
        # Attitude gains
        self.kR = torch.tensor([3000.0, 3000.0, 200.0], device=device) * self.cfg.moment_scale  # Rotation matrix error gain
        self.kw = torch.tensor([200.0, 200.0, 50.0], device=device) * self.cfg.moment_scale    # Angular velocity error gain
        
        # Gravity vector
        self.g = torch.tensor([0.0, 0.0, 9.81], device=device)
        self.robot_mass = 0.028  # Approximate Crazyflie mass (overwritten in env init)
        self.robot_weight = self.robot_mass * 9.81 # W = mg

    def compute_control(self, 
                        curr_pos: torch.Tensor, 
                        curr_vel: torch.Tensor, 
                        curr_quat: torch.Tensor, 
                        curr_ang_vel: torch.Tensor, 
                        target_vel: torch.Tensor, 
                        target_yaw_rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute control inputs (thrust, moments) to track target velocity and yaw rate.
        
        Args:
            curr_pos: (num_envs, 3) Current position in world frame
            curr_vel: (num_envs, 3) Current linear velocity in world frame
            curr_quat: (num_envs, 4) Current orientation (w, x, y, z)
            curr_ang_vel: (num_envs, 3) Current angular velocity in body frame
            target_vel: (num_envs, 3) Desired linear velocity in world frame
            target_yaw_rate: (num_envs, 1) Desired yaw rate
            
        Returns:
            thrust: (num_envs, 1, 3) Computed thrust force vector
            moments: (num_envs, 1, 3) Computed body torque vector
        """
        
        # 1. Compute Desired Force Vector (in World Frame)
        # error_v = v_des - v_curr
        # F_des = m * (g + K_v * error_v)
        
        # Velocity error
        # target_vel is in world frame (from policy)
        vel_error = target_vel - curr_vel
        
        # Desired acceleration
        acc_des = self.kv * vel_error + self.g
        
        # Desired force vector (F = ma)
        # Clamp acceleration to realistic limits to avoid instability
        acc_des_norm = torch.norm(acc_des, dim=1, keepdim=True)
        max_acc = 20.0 # m/s^2 limit
        acc_des = torch.where(acc_des_norm > max_acc, acc_des * (max_acc / acc_des_norm), acc_des)
        
        F_des = acc_des * self.robot_mass
        
        # 2. Compute Desired Attitude (Rotation Matrix)
        # The body z-axis (b3) should align with F_des
        b3_des = F_des / torch.norm(F_des, dim=1, keepdim=True)
        
        # Current rotation matrix R
        # quat is (w, x, y, z) in Isaac Lab
        w, x, y, z = curr_quat.unbind(-1)
        R_curr = torch.stack([
            1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
        ], dim=-1).reshape(self.num_envs, 3, 3)
        
        # 3. Compute Thrust Magnitude
        # Project desired force onto CURRENT body z-axis
        b3_curr = R_curr[:, :, 2]
        thrust_mag = torch.sum(F_des * b3_curr, dim=1, keepdim=True)
        # Clamp thrust to legitimate physical limits (0 to ~2.5x weight)
        # Dynamic clamping based on actual robot weight
        max_thrust = 2.5 * self.robot_weight
        thrust_mag = torch.clamp(thrust_mag, 0.0, max_thrust)
        
        thrust_vec = torch.zeros_like(F_des)
        thrust_vec[:, 2] = thrust_mag.squeeze(-1) # Force in body frame (z-axis)
        
        # 4. Compute Moments
        # We need to rotate b3_curr to b3_des
        # Cross product gives the axis of rotation needed
        
        # Rotation error vector (e_R)
        # e_R = 1/2 * (R_des.T * R - R.T * R_des)_vee ... standard geometric control
        # Simplified:
        # Target z-axis is b3_des. Current is b3_curr.
        # Orientation error to align z-axes:
        z_error = torch.linalg.cross(b3_curr, b3_des)
        
        # Rotate error to body frame!
        # z_error is in World Frame (because b3_curr and b3_des are World Frame vectors)
        # We need torque in Body Frame.
        # R_curr maps Body -> World. R_curr.T maps World -> Body.
        z_error_b = torch.matmul(R_curr.transpose(-2, -1), z_error.unsqueeze(-1)).squeeze(-1)
        
        # Yaw control:
        # We want angular velocity z component to match target_yaw_rate
        # We don't enforce a specific heading (yaw angle) because it's relative
        # So we just add a P-term on yaw rate error to the moment
        
        # Construct e_R (approximate)
        # The cross product z_error captures pitch/roll error
        e_R = z_error_b
        
        # Angular velocity error
        # We want body rates w_x, w_y to correct attitude, w_z to track yaw rate
        # Target w_z = target_yaw_rate
        # Target w_x, w_y = 0 (stabilize)
        
        target_ang_vel = torch.zeros_like(curr_ang_vel)
        target_ang_vel[:, 2] = target_yaw_rate.squeeze(-1)
        
        ang_vel_error = curr_ang_vel - target_ang_vel
        
        # Compute Moments
        # We separate tilt (roll/pitch) from yaw
        moments = -self.kR * e_R - self.kw * ang_vel_error
        
        # Clamp moments to avoid instability
        # For Crazyflie scale: Ixx ≈ 1.4e-5 kg⋅m², max angular accel ≈ 3000 rad/s²
        # Max torque ≈ 0.042 Nm. Use 0.5 Nm as safe upper bound (allows aggressive recovery)
        moments = torch.clamp(moments, -0.5, 0.5)  # FIX #2: Increased from ±0.05 to ±0.5
        
        return thrust_vec.unsqueeze(1), moments.unsqueeze(1)


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    """Configuration for GPS-denied quadcopter navigation environment."""
    
    # Environment settings
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 4  # (vx, vy, vz, yaw_rate)
    # Observation: lin_vel(3) + ang_vel(3) + gravity(3) + rel_goal(3) + goal_dist(1) + lidar_6(6) = 19
    observation_space: int = 19
    state_space: int = 0
    debug_vis: bool = True

    ui_window_class_type = QuadcopterEnvWindow

    # === CONTROLLER LIMITS ===
    max_target_velocity: float = 0.5  # m/s (reduced from 1.5 for stability)
    max_target_yaw_rate: float = 1.0  # rad/s (reduced from 2.0 for stability)

    # Physics simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # Robot configuration
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.1  # FIX #3: Increased from 0.01 (10x more attitude authority)

    # === SENSOR PARAMETERS ===
    # LiDAR / Depth sensor (simplified to 6-direction raycasting)
    lidar_num_rays: int = 6  # 6 directions: forward, back, left, right, up, down
    lidar_range_max: float = 2.0  # Maximum detection range (m)
    lidar_noise_std: float = 0.02  # Range measurement noise (m)

    # VIO (Visual-Inertial Odometry) - Noisy position/velocity estimation
    vio_enabled: bool = True
    vio_pos_drift_std: float = 0.05  # Cumulative position drift (m)
    vio_vel_noise_std: float = 0.02  # Velocity estimate noise (m/s)

    # === DISTURBANCE PARAMETERS ===
    # Wind disturbances
    wind_enabled: bool = True
    wind_random_force_max: float = 0.5  # Max wind force component (N)
    wind_freq: float = 0.5  # Wind change frequency (Hz)

    # === CONSTRAINT LIMITS ===
    max_velocity: float = 2.0  # m/s
    max_acceleration: float = 5.0  # m/s^2
    min_altitude: float = 0.05  # m (relaxed from 0.1 for learning)
    max_altitude: float = 3.0  # m (relaxed from 2.0 for headroom)
    max_tilt_angle: float = 1.5  # rad (~85 degrees, relaxed for recovery)

    # === OBSTACLE PARAMETERS ===
    num_obstacles: int = 0  # FIX #4: DISABLED for Phase 1 (learn hovering first, re-enable after iter 100)
    obstacle_radius_range: tuple = (0.1, 0.3)  # (min, max) radius
    obstacle_height_range: tuple = (0.5, 1.8)  # (min, max) height for obstacles (m)
    obstacle_speed_range: tuple = (0.0, 0.5)  # (min, max) linear velocity
    obstacle_collision_radius: float = 0.5  # Collision detection radius
    # === REWARD SCALING ===
    # Priority-based reward structure: survival > stability > goal reaching > smooth motion
    alive_reward_scale: float = 2.0       # Per-step reward for staying alive (increased to 2.0)
    upright_reward_scale: float = 8.0     # Reward for staying level (boosted: #1 priority for hover learning)
    distance_to_goal_scale: float = 2.0   # Dense reward for approaching goal (reduced from 15.0 to prioritize stability)
    collision_penalty_scale: float = 5.0   # Collision penalty (reduced from 10.0, less harsh early)
    velocity_penalty_scale: float = 0.1   # Penalize excessive velocity (small factor)
    acceleration_penalty_scale: float = 0.005  # FIX #5: Reduced from 0.01 (allow jerky motion during learning)


class QuadcopterEnv(DirectRLEnv):
    """GPS-denied quadcopter navigation environment with RL training."""
    
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action and control
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Initialize Geometric Controller
        self.controller = GeometricController(self.cfg, self.num_envs, self.device)
        
        # Goal position (world frame)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Obstacle positions and velocities (world frame)
        self._obstacle_pos_w = torch.zeros(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        self._obstacle_vel_w = torch.zeros(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        self._obstacle_radius = torch.zeros(self.num_envs, self.cfg.num_obstacles, device=self.device)
        
        # Wind disturbance
        self._wind_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._wind_counter = torch.zeros(self.num_envs, device=self.device)
        
        # VIO/Odometry noise
        self._vio_position_error = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Previous velocity for acceleration computation
        self._prev_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Collision tracking
        self._in_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_distances = torch.ones(self.num_envs, self.cfg.num_obstacles, device=self.device) * 10.0

        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "upright",
                "distance_to_goal",
                "collision_penalty",
                "velocity_penalty",
                "acceleration_penalty",
            ]
        }
        
        # Robot properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        
        # Update Controller Mass
        self.controller.robot_mass = self._robot_mass.item()
        self.controller.robot_weight = self._robot_weight

        print(f"[QuadcopterEnv] Robot Mass: {self._robot_mass.item():.4f} kg")
        print(f"[QuadcopterEnv] Robot Weight: {self._robot_weight:.4f} N")
        print(f"[QuadcopterEnv] Max Thrust (Clamped): {2.5 * self._robot_weight:.4f} N")

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """Setup the simulation scene with robot, terrain, and obstacles."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Add dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step.
        
        Actions are now high-level commands: [target_vx, target_vy, target_vz, target_yaw_rate]
        The Geometric Controller converts these to thrust and moments.
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # Interpret actions as velocity targets
        # actions[0:3]: target velocity (vx, vy, vz) in world frame, scale to max limits
        # actions[3]: target yaw rate
        target_vel = self._actions[:, :3] * self.cfg.max_target_velocity
        target_yaw_rate = self._actions[:, 3:4] * self.cfg.max_target_yaw_rate
        
        # Get current state
        curr_pos = self._robot.data.root_pos_w
        curr_vel = self._robot.data.root_lin_vel_w  # World frame velocity for controller
        curr_quat = self._robot.data.root_quat_w
        curr_ang_vel = self._robot.data.root_ang_vel_b
        
        # Compute Low-Level Control
        thrust, moments = self.controller.compute_control(
            curr_pos, curr_vel, curr_quat, curr_ang_vel, target_vel, target_yaw_rate
        )
        
        self._thrust = thrust
        self._moment = moments
        
        # Update wind disturbance (DISABLED for stability baseline)
        # self._update_wind_disturbance()
        
        # Update obstacle positions (circular trajectories)
        self._update_obstacles()
        
        # Update VIO error (cumulative drift) (DISABLED for stability baseline)
        # self._update_vio_error()

    def _apply_action(self):
        """Apply computed forces and torques to the robot."""
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )
        
        # Apply wind disturbance (DISABLED for stability baseline)
        # if self.cfg.wind_enabled:
        #     self._robot.permanent_wrench_composer.set_forces_and_torques(
        #         body_ids=self._body_id, forces=self._wind_force.unsqueeze(1)
        #     )

    def _get_observations(self) -> dict:
        """Get observation vector for policy.
        
        Observation includes (all normalized to similar scales):
        - Linear velocity in body frame (3) - normalized to ~[-1, 1]
        - Angular velocity in body frame (3) - normalized to ~[-1, 1]
        - Projected gravity in body frame (3) - already normalized
        - Relative goal position in body frame (3) - normalized to ~[-1, 1]
        - Distance to goal (1) - normalized to [0, 1]
        - LiDAR distances in 6 directions (6) - already normalized
        Total: 19 dimensions
        """
        # Relative goal position in body frame
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        
        # Distance to goal (normalized)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1, keepdim=True)
        distance_to_goal_normalized = torch.clamp(distance_to_goal / 5.0, 0, 1)  # Normalize to [0, 1]
        
        # LiDAR-like obstacle detection (simplified)
        lidar_obs = self._compute_lidar_observations()
        
        # MANUAL NORMALIZATION: Scale all observations to similar ranges [-1, 1] or [0, 1]
        # This helps the neural network learn faster by putting all inputs on the same scale
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b / 2.0,        # (3) Normalize: max expected ~2 m/s
                self._robot.data.root_ang_vel_b / 10.0,       # (3) Normalize: max expected ~10 rad/s
                self._robot.data.projected_gravity_b,         # (3) Already normalized
                desired_pos_b / 5.0,                          # (3) Normalize: max distance ~5m
                distance_to_goal_normalized,                  # (1) Already normalized
                lidar_obs,                                    # (6) Already normalized
            ],
            dim=-1,
        )
        
        # Verify and handle numerical issues (NaNs/Infs) due to potential physics instability
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            # Replace NaNs with 0 and Infs with finite values
            obs = torch.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
            
        return {"policy": obs}

    def _compute_lidar_observations(self) -> torch.Tensor:
        """Compute simplified LiDAR observations (6 directions)."""
        distances = torch.ones(self.num_envs, 6, device=self.device) * self.cfg.lidar_range_max
        max_range = self.cfg.lidar_range_max
        
        # Check specific obstacle distances (simplified ray casting)
        # We calculate vector to each obstacle, project onto 6 axes
        
        for obs_idx in range(self.cfg.num_obstacles):
            # Vector to obstacle in body frame
            rel_pos_w = self._obstacle_pos_w[:, obs_idx, :] - self._robot.data.root_pos_w
            rel_pos_b, _ = subtract_frame_transforms(
                torch.zeros_like(rel_pos_w), self._robot.data.root_quat_w, rel_pos_w
            )
            
            # Euclidean distance
            dist = torch.linalg.norm(rel_pos_b, dim=1)
            
            # Assign to closest direction (ray casting simplified)
            # For each obstacle, find which direction it's in and update if closer
            dist_forward = torch.abs(rel_pos_b[:, 0])  # Distance along body-x
            dist_right = torch.abs(rel_pos_b[:, 1])    # Distance along body-y
            dist_up = torch.abs(rel_pos_b[:, 2])       # Distance along body-z
            
            # Update distances (keep minimum)
            distances[:, 0] = torch.minimum(distances[:, 0], torch.where(rel_pos_b[:, 0] > 0, dist, torch.tensor(max_range, device=self.device)))  # forward
            distances[:, 1] = torch.minimum(distances[:, 1], torch.where(rel_pos_b[:, 0] < 0, dist, torch.tensor(max_range, device=self.device)))  # back
            distances[:, 2] = torch.minimum(distances[:, 2], torch.where(rel_pos_b[:, 1] > 0, dist, torch.tensor(max_range, device=self.device)))  # right
            distances[:, 3] = torch.minimum(distances[:, 3], torch.where(rel_pos_b[:, 1] < 0, dist, torch.tensor(max_range, device=self.device)))  # left
            distances[:, 4] = torch.minimum(distances[:, 4], torch.where(rel_pos_b[:, 2] > 0, dist, torch.tensor(max_range, device=self.device)))  # up
            distances[:, 5] = torch.minimum(distances[:, 5], torch.where(rel_pos_b[:, 2] < 0, dist, torch.tensor(max_range, device=self.device)))  # down
        
        # Normalize to [0, 1] (noise DISABLED for stability baseline)
        normalized_distances = distances / max_range
        # noise = torch.randn_like(normalized_distances) * self.cfg.lidar_noise_std
        # normalized_distances = torch.clamp(normalized_distances + noise, 0, 1)
        normalized_distances = torch.clamp(normalized_distances, 0, 1)
        
        return normalized_distances

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward based on priority: survival > stability > goal reaching > smooth motion."""
        
        # 1. ALIVE REWARD (survival bonus - most important early signal)
        alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward_scale * self.step_dt
        
        # 2. UPRIGHT REWARD (positive framing: reward being level)
        # When perfectly upright: projected_gravity_b = (0, 0, -1), so z = -1
        # -z gives 1.0 when upright, -1.0 when inverted
        upright = -self._robot.data.projected_gravity_b[:, 2]
        upright_reward = torch.clamp(upright, 0, 1) * self.cfg.upright_reward_scale * self.step_dt
        
        # 3. Distance to goal (primary navigation objective - dense reward)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # Smooth mapping: closer = higher reward, asymptotic at goal
        distance_to_goal_reward = (1.0 - torch.tanh(distance_to_goal / 0.8)) * self.cfg.distance_to_goal_scale * self.step_dt
        
        # 4. Collision penalty
        collision_penalty = torch.zeros(self.num_envs, device=self.device)
        for obs_idx in range(self.cfg.num_obstacles):
            rel_pos_w = self._obstacle_pos_w[:, obs_idx, :] - self._robot.data.root_pos_w
            dist_to_obs = torch.linalg.norm(rel_pos_w, dim=1)
            min_distance = self._obstacle_radius[:, obs_idx] + 0.25  # Robot safety radius
            collision_mask = dist_to_obs < min_distance
            collision_penalty = torch.where(
                collision_mask,
                collision_penalty + self.cfg.collision_penalty_scale,
                collision_penalty
            )
            self._in_collision = torch.logical_or(self._in_collision, collision_mask)
        
        # 5. Velocity penalty (penalize excessive speed)
        lin_vel_magnitude = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        velocity_penalty = torch.clamp(lin_vel_magnitude - self.cfg.max_velocity, 0) * self.cfg.velocity_penalty_scale * self.step_dt
        
        # 6. Acceleration penalty (promote smooth motion)
        lin_acc_b = (self._robot.data.root_lin_vel_b - self._prev_lin_vel_b) / self.step_dt
        lin_acc_magnitude = torch.linalg.norm(lin_acc_b, dim=1)
        acceleration_penalty = lin_acc_magnitude * self.cfg.acceleration_penalty_scale * self.step_dt
        
        # Combine rewards (positive rewards + negative penalties)
        rewards = {
            "alive": alive_reward,
            "upright": upright_reward,
            "distance_to_goal": distance_to_goal_reward,
            "collision": -collision_penalty,
            "velocity": -velocity_penalty,
            "acceleration": -acceleration_penalty,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Log episode sums
        self._episode_sums["alive"] += rewards["alive"]
        self._episode_sums["upright"] += rewards["upright"]
        self._episode_sums["distance_to_goal"] += rewards["distance_to_goal"]
        self._episode_sums["collision_penalty"] += rewards["collision"]
        self._episode_sums["velocity_penalty"] += rewards["velocity"]
        self._episode_sums["acceleration_penalty"] += rewards["acceleration"]
        
        # FIX #6: Update previous velocity for next step's acceleration computation (was missing!)
        self._prev_lin_vel_b = self._robot.data.root_lin_vel_b.clone()
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Altitude constraints
        below_min = self._robot.data.root_pos_w[:, 2] < self.cfg.min_altitude
        above_max = self._robot.data.root_pos_w[:, 2] > self.cfg.max_altitude
        
        # Tilt constraint (tip over detection)
        # Extract roll and pitch from quaternion (simplified check)
        gravity_error = torch.linalg.norm(self._robot.data.projected_gravity_b[:, :2], dim=1)
        tipped = gravity_error > 1.2  # ~69 degrees tilt (relaxed from 0.9 for recovery time)
        
        died = torch.logical_or(torch.logical_or(below_min, above_max), tipped)
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Log metrics
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # Reset robot state
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset state variables
        self._actions[env_ids] = 0.0
        self._prev_lin_vel_b[env_ids] = 0.0
        self._vio_position_error[env_ids] = 0.0
        self._wind_counter[env_ids] = 0.0
        self._in_collision[env_ids] = False
        
        # Sample new goal positions (relative to environment origin, in local xy plane)
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Sample obstacle positions and velocities
        self._sample_obstacles(env_ids)
        
        # Reset robot pose
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # FIX #7: Spawn at safe hover altitude (0.5m) instead of ground level (prevents instant death)
        default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + 0.5
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _sample_obstacles(self, env_ids: torch.Tensor):
        """Sample random obstacle positions and velocities."""
        for obs_idx in range(self.cfg.num_obstacles):
            # Random position in xy plane, fixed height
            xy_pos = torch.rand(len(env_ids), 2, device=self.device) * 4.0 - 2.0
            xy_pos += self._terrain.env_origins[env_ids, :2]
            
            # Ensure obstacles are at least 0.8m from spawn origin (safe zone)
            rel_xy = xy_pos - self._terrain.env_origins[env_ids, :2]
            dist_from_origin = torch.linalg.norm(rel_xy, dim=1, keepdim=True).clamp(min=0.01)
            too_close = (dist_from_origin < 0.8).squeeze(-1)
            if too_close.any():
                # Push outward to 0.8m minimum distance
                xy_pos[too_close] = self._terrain.env_origins[env_ids[too_close], :2] + rel_xy[too_close] * (0.8 / dist_from_origin[too_close])
            height = torch.rand(len(env_ids), 1, device=self.device) * (self.cfg.obstacle_height_range[1] - self.cfg.obstacle_height_range[0]) + self.cfg.obstacle_height_range[0]
            
            self._obstacle_pos_w[env_ids, obs_idx, :2] = xy_pos
            self._obstacle_pos_w[env_ids, obs_idx, 2] = height.squeeze(-1)
            
            # Random radius
            self._obstacle_radius[env_ids, obs_idx] = (
                torch.rand(len(env_ids), device=self.device) * 
                (self.cfg.obstacle_radius_range[1] - self.cfg.obstacle_radius_range[0]) + 
                self.cfg.obstacle_radius_range[0]
            )
            
            # Random velocity (circular/randomized trajectories)
            speed = torch.rand(len(env_ids), device=self.device) * (self.cfg.obstacle_speed_range[1] - self.cfg.obstacle_speed_range[0]) + self.cfg.obstacle_speed_range[0]
            angle = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
            self._obstacle_vel_w[env_ids, obs_idx, 0] = speed * torch.cos(angle)
            self._obstacle_vel_w[env_ids, obs_idx, 1] = speed * torch.sin(angle)

    def _update_obstacles(self):
        """Update obstacle positions (move in xy plane)."""
        self._obstacle_pos_w[:, :, :2] += self._obstacle_vel_w[:, :, :2] * self.sim.cfg.dt

    def _update_wind_disturbance(self):
        """Update wind disturbance with low-frequency variation."""
        self._wind_counter += self.sim.cfg.dt
        update_freq = self.cfg.wind_freq
        
        # Change wind every 1/update_freq seconds
        change_mask = self._wind_counter > (1.0 / update_freq)
        if change_mask.any():
            random_force = torch.randn(self.num_envs, 3, device=self.device) * self.cfg.wind_random_force_max
            self._wind_force = torch.where(change_mask.unsqueeze(-1), random_force, self._wind_force)
            self._wind_counter = torch.where(change_mask, torch.zeros_like(self._wind_counter), self._wind_counter)

    def _update_vio_error(self):
        """Update VIO cumulative position error (drift simulation)."""
        # Random walk in position error
        drift_noise = torch.randn(self.num_envs, 3, device=self.device) * self.cfg.vio_pos_drift_std * self.sim.cfg.dt
        self._vio_position_error += drift_noise
        # Clamp to prevent unbounded drift
        self._vio_position_error = torch.clamp(self._vio_position_error, -0.5, 0.5)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            
            if not hasattr(self, "obstacle_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/obstacles"
                self.obstacle_visualizer = VisualizationMarkers(marker_cfg)
            
            self.goal_pos_visualizer.set_visibility(True)
            self.obstacle_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "obstacle_visualizer"):
                self.obstacle_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization markers."""
        # Update goal position
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        
        # FIX #8: Only visualize obstacles if they exist (prevents error when num_obstacles=0)
        if self.cfg.num_obstacles > 0:
            obs_positions_flat = self._obstacle_pos_w.view(-1, 3)
            self.obstacle_visualizer.visualize(obs_positions_flat)
