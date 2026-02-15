# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained RSL-RL policy in a live urban-fire demo scene."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Live urban-fire demo with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--hybrid_edit_mode",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="When paused in the timeline, stop policy stepping so scene editing can continue visually.",
)

# Scene knobs
parser.add_argument("--fire_scene_scale", type=float, default=1.0, help="Global XY scale for the urban scene.")
parser.add_argument("--disable_fire_scene", action="store_true", default=False, help="Disable custom urban-fire scene.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Sensible default for remote demo if user didn't set one
if args_cli.livestream == -1:
    args_cli.livestream = 2

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

try:
    import omni.timeline
except ImportError:
    omni = None


def spawn_urban_fire_scene(scale: float):
    """Spawn static buildings + fire spots across env instances via regex prim paths."""

    # Spawn directly under each env root since /World/envs/env_.* already exists.
    # Using a nested path like /LiveDemo requires explicitly creating that prim first.
    env_path = "/World/envs/env_.*"

    road_cfg = sim_utils.CuboidCfg(
        size=(7.0 * scale, 7.0 * scale, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12), roughness=0.95),
    )
    road_cfg.func(f"{env_path}/RoadPatch", road_cfg, translation=(0.0, 0.0, 0.01))

    buildings = [
        (-2.2, -1.8, 1.0, 1.0, 2.2),
        (-2.1, 1.6, 0.9, 1.3, 2.8),
        (2.0, -1.9, 1.2, 0.9, 2.5),
        (2.1, 1.8, 1.1, 1.1, 3.0),
        (0.0, 2.3, 1.6, 0.7, 2.0),
        (0.0, -2.3, 1.6, 0.7, 2.0),
    ]
    for idx, (x, y, sx, sy, h) in enumerate(buildings):
        building_cfg = sim_utils.CuboidCfg(
            size=(sx * scale, sy * scale, h),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.23, 0.25), roughness=0.9, metallic=0.05),
        )
        building_cfg.func(
            f"{env_path}/Building_{idx}",
            building_cfg,
            translation=(x * scale, y * scale, h * 0.5),
        )

    fire_points = [(-1.0, 0.0), (0.9, -0.7), (1.2, 1.0)]
    for idx, (x, y) in enumerate(fire_points):
        fire_cfg = sim_utils.SphereCfg(
            radius=0.14,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.35, 0.08),
                emissive_color=(1.0, 0.45, 0.1),
                roughness=0.2,
            ),
        )
        fire_cfg.func(
            f"{env_path}/Fire_{idx}",
            fire_cfg,
            translation=(x * scale, y * scale, 0.14),
        )

        smoke_cfg = sim_utils.CylinderCfg(
            radius=0.12,
            height=0.9,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.15),
                opacity=0.25,
                roughness=1.0,
            ),
        )
        smoke_cfg.func(
            f"{env_path}/Smoke_{idx}",
            smoke_cfg,
            translation=(x * scale, y * scale, 0.6),
        )

        fire_light_cfg = sim_utils.SphereLightCfg(
            color=(1.0, 0.45, 0.2),
            intensity=25000.0,
            radius=0.35,
        )
        fire_light_cfg.func(
            f"{env_path}/FireLight_{idx}",
            fire_light_cfg,
            translation=(x * scale, y * scale, 0.55),
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # spawn urban scene only for demo
    if not args_cli.disable_fire_scene:
        spawn_urban_fire_scene(args_cli.fire_scene_scale)
        print("[INFO] Spawned urban-fire demo scene.")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    use_hybrid_edit_mode = args_cli.hybrid_edit_mode and omni is not None and args_cli.livestream == 0
    if args_cli.hybrid_edit_mode and args_cli.livestream > 0:
        print("[WARN] Hybrid edit mode is disabled during WebRTC livestream to avoid stream timeouts.")
    timeline = omni.timeline.get_timeline_interface() if use_hybrid_edit_mode else None
    if timeline is not None:
        print("[INFO] Hybrid edit mode enabled (GUI): pause/play in Isaac Sim timeline to edit and resume.")
    # simulate environment
    while simulation_app.is_running():
        if timeline is not None and not timeline.is_playing():
            time.sleep(0.05)
            continue

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
