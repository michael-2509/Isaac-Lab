# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.utils.math import subtract_frame_transforms, quat_rotate

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, SPHERE_MARKER_CFG  # isort: skip

# For random obstacle generation
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas.schemas_physics import RigidBodyPropertiesCfg


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    # Updated observation space: 
    # - lin_vel_b (3) + ang_vel_b (3) + projected_gravity_b (3) + relative_goal_b (3) 
    # + distance_to_goal (1) + lidar_distances (6) = 19
    observation_space = 19
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # Simulation
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
    terrain = TerrainImporterCfg(
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Sensor parameters
    lidar_num_rays = 32  # Number of LiDAR rays
    lidar_range_min = 0.1  # Minimum range (m)
    lidar_range_max = 4.0  # Maximum range (m)
    lidar_noise_std = 0.01  # Range measurement noise

    # Odometry noise (VIO-like)
    odometry_pos_noise_std = 0.05  # Position estimate noise
    odometry_vel_noise_std = 0.02  # Velocity estimate noise

    # Wind disturbance
    wind_enabled = True
    wind_random_force_max = 0.5  # Max random force component (N)
    wind_freq = 0.5  # Wind change frequency (Hz)

    # Constraint limits
    max_velocity = 2.0  # m/s
    max_acceleration = 5.0  # m/s^2
    min_altitude = 0.1  # m
    max_altitude = 2.0  # m

    # Obstacle parameters
    num_obstacles = 4
    obstacle_radius_range = (0.1, 0.3)
    obstacle_height_range = (0.2, 0.8)

    # Reward scales
    distance_to_goal_scale = 15.0
    collision_penalty_scale = 10.0
    velocity_penalty_scale = -0.05
    acceleration_penalty_scale = -0.02
    orientation_penalty_scale = -0.01
    min_height = 0.1  # m
    
    # wind disturbances
    wind_enabled = True
    wind_mean = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    wind_scale = 0.3  # std dev of wind disturbance
    
    # VIO/Odometry noise
    vio_enabled = True
    position_noise_scale = 0.02  # meters
    velocity_noise_scale = 0.01  # m/s
    
    # obstacles
    num_obstacles = 3
    obstacle_radius = 0.3  # meters
    obstacle_collision_distance = 0.5  # detection range
    
    # reward scales (priority-based)
    distance_to_goal_reward_scale = 15.0  # Dense reward for reaching goal
    collision_penalty_scale = -50.0  # Large penalty for collision
    lin_vel_reward_scale = -0.05  # Penalize linear velocity
    ang_vel_reward_scale = -0.01  # Penalize angular velocity
    acceleration_penalty_scale = -0.005  # Penalize large accelerations
    orientation_stability_scale = -0.01  # Reward for staying upright


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Goal position (in world frame)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Obstacle positions (world frame)
        self._obstacle_pos_w = torch.zeros(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        self._obstacle_vel_w = torch.zeros(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        
        # Wind disturbances
        self._wind_disturbance = torch.zeros(self.num_envs, 3, device=self.device)
        
        # VIO noise (cumulative position error)
        self._vio_position_error = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Previous actions for acceleration penalty
        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "collision",
                "acceleration",
                "orientation_stability",
            ]
        }
        
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions with safety constraints."""
        # Clamp actions and save for acceleration penalty
        self._last_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # Update wind disturbance (simple random walk)
        if self.cfg.wind_enabled:
            self._wind_disturbance += torch.randn_like(self._wind_disturbance) * self.cfg.wind_scale
            # Low-pass filter for smoothness
            self._wind_disturbance = 0.9 * self._wind_disturbance + 0.1 * torch.randn_like(self._wind_disturbance) * self.cfg.wind_scale
        
        # Update VIO/odometry noise (cumulative)
        if self.cfg.vio_enabled:
            self._vio_position_error += torch.randn_like(self._vio_position_error) * self.cfg.position_noise_scale
        
        # Compute thrust with safety constraints
        # Scale thrust: [0, 1] action to [0, max_thrust]
        thrust_magnitude = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        
        # Apply thrust in world Z direction
        self._thrust[:, 0, 2] = thrust_magnitude
        
        # Apply torque moments for attitude control
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        
        # Add wind disturbance as perturbation (weak effect)
        self._thrust[:, 0, :2] += self._wind_disturbance[:, :2] * 0.1

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )
    
    def _step_sim(self, action: torch.Tensor) -> None:
        """Override step to update obstacle positions."""
        # Update obstacle positions (simple linear motion)
        self._obstacle_pos_w += self._obstacle_vel_w * self.sim.cfg.dt
        
        # Bounce obstacles off environment boundaries
        env_radius = 2.0  # approximate environment radius
        for i in range(self.cfg.num_obstacles):
            # Bounce in xy-plane
            bounce_mask = torch.abs(self._obstacle_pos_w[:, i, :2]) > env_radius
            self._obstacle_vel_w[:, i, :2][bounce_mask] *= -1.0
            self._obstacle_pos_w[:, i, :2] = torch.clamp(
                self._obstacle_pos_w[:, i, :2], -env_radius, env_radius
            )
            
            # Bounce in z (height)
            bounce_z_high = self._obstacle_pos_w[:, i, 2] > self.cfg.max_height
            bounce_z_low = self._obstacle_pos_w[:, i, 2] < self.cfg.min_height
            self._obstacle_vel_w[:, i, 2][bounce_z_high | bounce_z_low] *= -1.0
            self._obstacle_pos_w[:, i, 2] = torch.clamp(
                self._obstacle_pos_w[:, i, 2], self.cfg.min_height, self.cfg.max_height
            )
        
        # Call parent step simulation
        super()._step_sim(action)

    def _get_observations(self) -> dict:
        """Get observations including noisy odometry and relative goal in body frame."""
        # Get desired position relative to robot in body frame
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        
        # Distance to goal
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1, keepdim=True)
        
        # Compute closest obstacle distances in body frame (simplified LiDAR)
        # Returns 6 values (distances to obstacles in 6 directions: forward, back, left, right, up, down)
        obstacle_distances = self._compute_obstacle_distances()
        
        # Construct observation: [lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + goal_rel_b(3) + goal_range(1) + lidar(6)]
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,  # 3
                self._robot.data.root_ang_vel_b,  # 3
                self._robot.data.projected_gravity_b,  # 3
                desired_pos_b,  # 3 (relative goal in body frame)
                distance_to_goal,  # 1 (range to goal)
                obstacle_distances,  # 6 (simplified LiDAR)
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    
    def _compute_obstacle_distances(self) -> torch.Tensor:
        """Compute simplified LiDAR-like distances to obstacles in 6 directions (body frame).
        
        Returns distances in body-aligned directions: [forward, back, left, right, up, down]
        """
        # Transform obstacle positions to body frame
        # For each obstacle, compute distance and project onto body-aligned directions
        max_range = 2.0  # max detection range
        default_value = max_range  # when no obstacle detected
        
        distances = torch.full(
            (self.num_envs, 6), default_value, dtype=torch.float32, device=self.device
        )
        
        for i in range(self.cfg.num_obstacles):
            # Relative position in world frame
            rel_pos_w = self._obstacle_pos_w[:, i, :] - self._robot.data.root_pos_w
            
            # Transform to body frame
            rel_pos_b = quat_rotate(
                torch.cat([self._robot.data.root_quat_w[:, [3]], 
                          self._robot.data.root_quat_w[:, :3]], dim=-1),
                rel_pos_w
            )
            
            # Compute distance to obstacle
            dist = torch.linalg.norm(rel_pos_b, dim=1)
            
            # Only consider if within detection range
            mask = dist < max_range
            
            # Assign to closest direction
            # Forward (x), Back (-x), Right (y), Left (-y), Up (z), Down (-z)
            abs_pos = torch.abs(rel_pos_b)
            max_idx = torch.argmax(abs_pos, dim=1)
            
            # Update distance for each direction
            for env_idx in range(self.num_envs):
                if mask[env_idx]:
                    if rel_pos_b[env_idx, 0] > 0:  # forward
                        distances[env_idx, 0] = torch.min(distances[env_idx, 0:1], dist[env_idx:env_idx+1])
                    else:  # back
                        distances[env_idx, 1] = torch.min(distances[env_idx, 1:2], dist[env_idx:env_idx+1])
                    
                    if rel_pos_b[env_idx, 1] > 0:  # right
                        distances[env_idx, 2] = torch.min(distances[env_idx, 2:3], dist[env_idx:env_idx+1])
                    else:  # left
                        distances[env_idx, 3] = torch.min(distances[env_idx, 3:4], dist[env_idx:env_idx+1])
                    
                    if rel_pos_b[env_idx, 2] > 0:  # up
                        distances[env_idx, 4] = torch.min(distances[env_idx, 4:5], dist[env_idx:env_idx+1])
                    else:  # down
                        distances[env_idx, 5] = torch.min(distances[env_idx, 5:6], dist[env_idx:env_idx+1])
        
        return distances

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with priority: reach goal > avoid collision > smooth motion > orientation."""
        # 1. Distance to goal (dense reward)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        
        # 2. Collision penalty (large negative reward)
        collision_penalty = self._compute_collision_penalty()
        
        # 3. Smooth motion penalties
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        
        # Acceleration penalty (difference from last action)
        action_diff = torch.norm(self._actions - self._last_actions, dim=1)
        
        # 4. Orientation stability (penalize tilt when not moving)
        # Gravity vector in body frame should point down: [0, 0, -1]
        gravity_alignment = torch.sum(torch.abs(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        rewards = {
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "collision": collision_penalty * self.cfg.collision_penalty_scale * self.step_dt,
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "acceleration": action_diff * self.cfg.acceleration_penalty_scale * self.step_dt,
            "orientation_stability": gravity_alignment * self.cfg.orientation_stability_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward
    
    def _compute_collision_penalty(self) -> torch.Tensor:
        """Compute collision penalty based on proximity to obstacles."""
        penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        for i in range(self.cfg.num_obstacles):
            # Distance to obstacle
            rel_pos = self._obstacle_pos_w[:, i, :] - self._robot.data.root_pos_w
            dist_to_obstacle = torch.linalg.norm(rel_pos, dim=1)
            
            # Penalty increases exponentially as we approach collision distance
            # penalty = 1 if within collision_distance, exponential falloff beyond
            collision_dist = self.cfg.obstacle_collision_distance
            collision_penalty = torch.clamp(
                torch.exp(-dist_to_obstacle / collision_dist), 0, 1
            )
            penalty = torch.max(penalty, collision_penalty)
        
        return penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions: timeout, height bounds, or collision."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Height constraints with margin
        height_out = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < self.cfg.min_height,
            self._robot.data.root_pos_w[:, 2] > self.cfg.max_height
        )
        
        # Collision with obstacles
        collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.cfg.num_obstacles):
            rel_pos = self._obstacle_pos_w[:, i, :] - self._robot.data.root_pos_w
            dist_to_obstacle = torch.linalg.norm(rel_pos, dim=1)
            collision = torch.logical_or(
                collision, 
                dist_to_obstacle < self.cfg.obstacle_radius
            )
        
        # Combined termination condition
        died = torch.logical_or(height_out, collision)
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with randomized goals and obstacles."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
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

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        
        # Sample new goal with randomized trajectory
        # Goals sampled in xy-plane around robot, with varying heights
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Sample moving obstacles with randomized trajectories
        for i in range(self.cfg.num_obstacles):
            # Random position in environment
            self._obstacle_pos_w[env_ids, i, :2] = torch.zeros_like(
                self._obstacle_pos_w[env_ids, i, :2]
            ).uniform_(-2.0, 2.0)
            self._obstacle_pos_w[env_ids, i, :2] += self._terrain.env_origins[env_ids, :2]
            self._obstacle_pos_w[env_ids, i, 2] = torch.zeros_like(
                self._obstacle_pos_w[env_ids, i, 2]
            ).uniform_(0.5, 1.5)
            
            # Random velocity for moving obstacles
            self._obstacle_vel_w[env_ids, i, :] = torch.zeros_like(
                self._obstacle_vel_w[env_ids, i, :]
            ).uniform_(-0.5, 0.5)
        
        # Reset odometry error
        self._vio_position_error[env_ids] = 0.0
        self._wind_disturbance[env_ids] = 0.0
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            
            if not hasattr(self, "obstacle_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = self.cfg.obstacle_radius
                # -- obstacles
                marker_cfg.prim_path = "/Visuals/Obstacles/positions"
                self.obstacle_visualizer = VisualizationMarkers(marker_cfg)
            
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.obstacle_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "obstacle_visualizer"):
                self.obstacle_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        
        # Update obstacle positions (flatten to num_envs * num_obstacles)
        obstacle_pos_flat = self._obstacle_pos_w.reshape(self.num_envs * self.cfg.num_obstacles, 3)
        self.obstacle_visualizer.visualize(obstacle_pos_flat)
