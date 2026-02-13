# GPS-Denied Autonomous Quadcopter Navigation

## Project Overview

This is a hierarchical RL-based navigation system for autonomous quadcopters in GPS-denied, dynamic environments. The system is designed for **real-world deployment** on platforms like the Crazyflie 2.1 and targets applications such as:

- Autonomous fire detection drones
- Autonomous delivery systems
- Search and rescue operations
- Infrastructure inspection

## Architecture: VLM → RL → Controller

Currently implemented: **RL Policy Layer**

```
┌─────────────────────────────────────────────────────┐
│ High-Level Task (Future: VLM/LLM)                   │
│ "Reach target location and detect fire"             │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ RL Policy (Current Implementation)                  │
│ - Learns navigation in dynamic environments         │
│ - Outputs 4D control commands                       │
│ - Handles collision avoidance & goal-seeking        │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ Low-Level Controller (Onboard)                      │
│ - Thrust allocation                                 │
│ - Attitude control                                  │
│ - Motor command generation                          │
└─────────────────────────────────────────────────────┘
```

## Key Features Implemented

### 1. **Realistic Sensor Simulation**

#### Visual-Inertial Odometry (VIO)
- Simulates cumulative position drift typical of SLAM/VIO systems
- Configurable noise in position and velocity estimates
- Bounded drift to represent integrated odometry error
- **Parameter**: `vio_pos_drift_std`, `vio_vel_noise_std`

#### Simplified LiDAR/Depth Camera
- 6-direction raycasting (forward, back, left, right, up, down)
- Detects distances to moving obstacles
- Adds realistic measurement noise
- **Parameters**: `lidar_range_max`, `lidar_noise_std`

#### Proprioceptive Sensors
- Linear/angular velocity (from IMU)
- Gravity vector (accelerometer)
- All measurements in body frame for realistic control

### 2. **Dynamic Environment**

#### Moving Obstacles
- Randomized trajectories (circular + random walk)
- Variable sizes and speeds
- Placed in continuous xy-plane motion
- **Configuration**:
  - `num_obstacles`: 4 dynamic obstacles
  - `obstacle_radius_range`: (0.1 - 0.3 m)
  - `obstacle_speed_range`: (0.0 - 0.5 m/s)

#### Environmental Disturbances

**Wind Model**:
- Low-frequency wind force variations
- Acts as external force disturbance
- Simulates turbulence at building-scale
- **Parameters**: `wind_enabled`, `wind_random_force_max`, `wind_freq`

### 3. **State Constraints (Safety)**

- **Altitude limits**: [0.1 m, 2.0 m]
- **Velocity limit**: 2.0 m/s
- **Acceleration limit**: 5.0 m/s²
- **Tilt constraint**: Up to 45° (prevents tip-over)
- **Episodic termination** when violated

### 4. **Priority-Based Reward Structure**

The reward function emphasizes stable, goal-reaching behavior:

| Objective | Priority | Scale | Purpose |
|-----------|----------|-------|---------|
| Distance to Goal | 1️⃣ Primary | +15.0 | Dense reward for progress |
| Collision Avoidance | 2️⃣ Secondary | -10.0 | Large penalty for obstacles |
| Velocity Penalty | 3️⃣ Tertiary | -0.05 | Encourage smooth motion |
| Acceleration Penalty | 4️⃣ | -0.02 | Reduce jerkiness |
| Orientation Penalty | 5️⃣ | -0.01 | Keep drone upright |

**Why this structure?**
- Dense goal reward ensures convergence
- Collision penalty prevents crashes
- Velocity/acceleration penalties improve real-world performance (reduces wear, improves control authority)
- Orientation penalty maintains stability

## Observation Space (19 dimensions)

```
[lin_vel_b(3), ang_vel_b(3), gravity_b(3), rel_goal_b(3), goal_dist(1), lidar(6)]
         ▲                                         ▲
         └─ Body-frame measurements    └─ Relative goal in body frame (local coordinates)
```

**Why body-frame observations?**
- Body-frame is invariant to world-frame orientation
- Matches onboard sensor outputs
- Enables transfer to real hardware

**Why relative goal?**
- GPS-denied → no absolute localization
- Goal is relative to VIO estimate
- Matches real deployment scenario

## Action Space (4 dimensions)

```
[thrust, roll_rate, pitch_rate, yaw_rate]
   ▲                      ▲
   └─ Vertical thrust    └─ Angular velocity commands
```

**Control model**:
- Thrust: 0-100% (thrust_to_weight = 1.9)
- Rates: ±max_rate (scaled by moment_scale)
- Actions clipped to [-1, 1] during execution

## Training Configuration

### hyperparameters (skrl_ppo_cfg.yaml)

```yaml
Algorithm: PPO (Proximal Policy Optimization)

Network Architecture:
  - Policy/Value networks: [128, 128] hidden layers
  - Activation: ELU
  - Output: Gaussian policy (continuous)

Training:
  - Rollouts per update: 32
  - Learning epochs: 5
  - Learning rate: 5.0e-4 (KL-adaptive)
  - Total timesteps: 100,000
  - Discount factor: 0.99
  - GAE lambda: 0.95
```

### Environment Configuration

**Parallelization**:
- 4096 parallel environments
- Environment spacing: 2.5 m
- Physics replication (fabric-based)

**Simulation**:
- Timestep: 10 ms (100 Hz)
- Decimation: 2 (policy runs at 50 Hz)

## File Structure

```
quadcopter/
├── quadcopter_env.py              # Main environment implementation
├── __init__.py                    # Environment registration
├── GPS_DENIED_NAVIGATION.md       # This file
└── agents/
    ├── skrl_ppo_cfg.yaml         # PPO training config
    ├── rl_games_ppo_cfg.yaml     # Alternative RL-Games config
    └── rsl_rl_ppo_cfg.py         # RSL-RL config
```

## How to Train

```bash
cd /home/ubuntu/IsaacLab

# Basic training (4096 parallel envs, GPU)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --agent skrl_cfg_entry_point

# With custom seeds and max iterations
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --seed 42 \
    --max_iterations 500 \
    --num_envs 2048

# For distributed training (multi-GPU)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Quadcopter-Direct-v0 \
    --distributed \
    --device cuda:0
```

## Expected Performance Metrics

### Early Training (10k steps)
- Reward: ~-5 to 0 (baseline crash-free behavior)
- Success: Learning collision avoidance

### Mid Training (50k steps)
- Reward: ~5 to 15 (consistent goal reaching)
- Success: Reaching 80% of goals without collision

### Late Training (100k+ steps)
- Reward: ~15+ (optimal behavior)
- Success: Reaching 95%+ goals with smooth trajectories

### Episode Logs
Training logs include:
```
Episode_Reward/distance_to_goal      # Progress toward goal
Episode_Reward/collision_penalty     # Collision frequency
Episode_Reward/velocity_penalty      # Speed control
Episode_Reward/acceleration_penalty  # Smoothness
Episode_Reward/orientation_penalty   # Stability
Episode_Termination/died            # Crash rate
Episode_Termination/time_out        # Natural episode end
Metrics/final_distance_to_goal      # Success distance
```

## Real-World Deployment Considerations

### 1. **Sim-to-Real Transfer**

**What transfers well:**
- ✅ Relative navigation (body-frame observations)
- ✅ Collision avoidance behaviors
- ✅ Smooth trajectory control

**What needs adaptation:**
- ❌ Absolute localization (use real VIO/SLAM)
- ❌ Camera intrinsics (calibrate to real camera)
- ❌ Wind model (environment-specific)

### 2. **Onboard Deployment (Crazyflie)**

**Computational constraints:**
- STM32F405 microcontroller (200 MHz)
- Limited inference capability → need model compression
- ~20 ms per policy evaluation acceptable

**Solution pathway:**
1. Train large network in sim (current: 128×128)
2. Export to ONNX
3. Quantize to INT8
4. Deploy on Crazyflie or edge device

### 3. **Sensor Integration**

**Current simulation assumptions:**
- Perfect VIO with bounded drift
- Simplified LiDAR (6 rays)

**Real hardware requirements:**
- **VIO**: Use ROS/SLAM backend (RTAB-Map, ORB-SLAM3)
- **LiDAR**: Use TOF sensor (VL53L1X) or depth camera
- **Integration**: ROS middleware for sensor→policy interface

### 4. **Safety Constraints**

**Implemented in training:**
- Altitude bounds (prevents crashing through floor/ceiling)
- Velocity limits (reduces inertial damage)
- Tilt constraints (prevents tip-over)

**Recommended for deployment:**
- Geofence enforcement (GPS fallback)
- Emergency landing on signal loss
- Watchdog timeout (auto-land after 2s no command)

## Research Extensions

### Phase 2: VLM Integration
- Input: High-res camera feed + semantic understanding
- Process: Vision→Language→Navigation goal
- Example: "Fly to the red building and check for fire"

### Phase 3: Robust Control
- Domain randomization for sim-to-real
- Adversarial training for wind robustness
- Curriculum learning (easy → hard navigation)

### Phase 4: Multi-Agent Coordination
- Multiple drones sharing airspace
- Centralized policy (current) → Decentralized MARL
- Communication protocol for coordination

## Troubleshooting

### Environment Crashes Immediately
```python
# Check constraint limits in QuadcopterEnvCfg
min_altitude = 0.1  # Must be > 0
max_altitude = 2.0  # Must be > min_altitude
max_velocity = 2.0  # Should match expected speeds
```

### Policy Doesn't Learn
```yaml
# In skrl_ppo_cfg.yaml:
learning_rate: 5.0e-04        # May need tuning
rollouts: 32                   # Increase for stability
learning_epochs: 5            # Increase for convergence
```

### Simulation Too Slow
```bash
# Reduce parallel environments
--num_envs 1024  # Default: 4096
--device cuda:0  # Ensure GPU usage
```

## References

- **IsaacLab**: https://github.com/isaac-sim/IsaacLab
- **SKRL**: https://skrl.readthedocs.io
- **Crazyflie**: https://www.bitcraze.io/crazyflie-2-1/
- **PPO Paper**: https://arxiv.org/abs/1707.06347

## Authors & Citation

```bibtex
@article{quadcopter_navigation_2026,
  title={GPS-Denied Autonomous Navigation for Quadcopters using Hierarchical RL},
  author={You},
  year={2026}
}
```

---

**Last Updated**: February 2026  
**Status**: Active Development - Ready for Phase 1 training
