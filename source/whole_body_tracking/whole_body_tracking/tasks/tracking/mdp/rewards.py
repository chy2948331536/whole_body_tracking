from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def motion_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_cfg: SceneEntityCfg,
    height_threshold: float = 0.05,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward for matching contact state with motion data foot height.
    
    If motion data foot height is below threshold, the foot should be in contact.
    Reward is given when actual contact state matches expected contact state.
    
    Args:
        env: The environment instance.
        command_name: Name of the motion command.
        contact_cfg: Configuration for the contact sensor with foot body names.
        height_threshold: Height threshold below which foot should be in contact (default: 0.05).
        force_threshold: Force threshold to determine if foot is in contact (default: 1.0).
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[contact_cfg.name]
    
    # Get foot body names from contact_cfg
    if isinstance(contact_cfg.body_names, str):
        foot_body_names = [contact_cfg.body_names]
    elif isinstance(contact_cfg.body_names, list):
        foot_body_names = contact_cfg.body_names
    else:
        # No body names specified, return zero reward
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    # Get foot body indexes in the command's body_names list
    foot_body_indexes = _get_body_indexes(command, foot_body_names)
    
    if len(foot_body_indexes) == 0:
        # No matching feet found, return zero reward
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    # Check if body_ids is available and is a list
    if contact_cfg.body_ids is None or not isinstance(contact_cfg.body_ids, list):
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    # Get motion data foot positions (world frame)
    # body_pos_w already includes env_origins, so Z coordinate is absolute
    # We need Z coordinate relative to ground (subtract env_origins Z)
    motion_foot_pos_w = command.body_pos_w[:, foot_body_indexes, :]  # (num_envs, num_feet, 3)
    env_origins_z = env.scene.env_origins[:, 2:3]  # (num_envs, 1)
    motion_foot_height = motion_foot_pos_w[:, :, 2] - env_origins_z  # (num_envs, num_feet)
    
    # Determine if foot should be in contact based on motion data height
    # is_stance = True if height < threshold (should be in contact)
    is_stance = motion_foot_height < height_threshold  # (num_envs, num_feet)
    # print(f"is_stance: {is_stance}")
    # Get contact forces from sensor
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # Initialize reward
    res = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    # Check each foot
    num_feet = len(foot_body_indexes)
    for i in range(num_feet):
        if i >= len(contact_cfg.body_ids):
            continue
        # Get contact state for this foot (Z component of force > threshold)
        # contact_cfg.body_ids[i] gives the body ID in the sensor
        body_id = contact_cfg.body_ids[i]
        contact = torch.max(
            net_contact_forces[:, :, body_id][:, :, 2],  # (num_envs, history_length)
            dim=1
        )[0] > force_threshold  # (num_envs,)
        
        # Reward when contact state matches expected state (XOR: ~(contact ^ is_stance))
        # This gives reward when: (contact=True and is_stance=True) or (contact=False and is_stance=False)
        res += (~(contact ^ is_stance[:, i])).float()
    
    return res
