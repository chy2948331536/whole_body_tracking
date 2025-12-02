from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Publish to ROS node if available
    if hasattr(env, 'termination_anchor_pos_z_node'):
        condition = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
        condition.to(torch.float)
        env.termination_anchor_pos_z_node.cb(
            condition.cpu().numpy().tolist(), 
            env.current_time.item() if hasattr(env, 'current_time') else None
        )

    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Publish to ROS node if available
    if hasattr(env, 'termination_anchor_pos_z_node'):
        condition = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
        condition.to(torch.float)
        env.termination_anchor_pos_z_node.cb(
            condition.cpu().numpy().tolist(), 
            env.current_time.item() if hasattr(env, 'current_time') else None
        )

    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    # Publish to ROS node if available
    if hasattr(env, 'termination_anchor_ori_node'):
        condition = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
        condition = condition.float()
        env.termination_anchor_ori_node.cb(
            condition.cpu().numpy().tolist(), 
            env.current_time.item() if hasattr(env, 'current_time') else None
        )
    
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    # Publish to ROS node if available
    if hasattr(env, 'termination_ee_body_pos_z_node'):
        # Publish max error and individual errors for each body
        error_data = torch.any(error > threshold, dim=-1).to(torch.float).cpu().numpy().tolist()
        env.termination_ee_body_pos_z_node.cb(
            error_data, 
            env.current_time.item() if hasattr(env, 'current_time') else None
        )

    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])

    # Publish to ROS node if available
    if hasattr(env, 'termination_ee_body_pos_z_node'):
        # Publish max error and individual errors for each body
        error_data = torch.any(error > threshold, dim=-1).to(torch.float).cpu().numpy().tolist()
        env.termination_ee_body_pos_z_node.cb(
            error_data, 
            env.current_time.item() if hasattr(env, 'current_time') else None
        )
    
    return torch.any(error > threshold, dim=-1)
