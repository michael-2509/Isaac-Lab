# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48  # Increased from 24 for larger batch size
    max_iterations = 200
    save_interval = 50
    experiment_name = "quadcopter_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,  # FIX #10: Reduced from 1.0 (gentler initial exploration)
        actor_obs_normalization=False,  # Using manual normalization in environment instead
        critic_obs_normalization=False,  # Using manual normalization in environment instead
        actor_hidden_dims=[128, 128, 64],  # Increased from [64, 64] for better representation
        critic_hidden_dims=[128, 128, 64],  # Increased from [64, 64] to match actor
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Increased from 0.0 for better exploration
        num_learning_epochs=5,
        num_mini_batches=32,  # Increased from 4 for more frequent gradient updates
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
