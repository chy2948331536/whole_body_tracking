# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pprint import pprint

import numpy as np
import torch

import onnx
import onnxruntime as ort

# Set numpy print options: 4 decimal places, no scientific notation
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute the conjugate of quaternion(s). Quaternion format: (w, x, y, z)."""
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions. Quaternion format: (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to rotation matrix. Quaternion format: (w, x, y, z)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return np.stack(
        [
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


class OnnxPolicy:
    """ONNX-based policy wrapper for inference (single-step, no history)."""

    def __init__(self, onnx_path: str, device: str = "cuda"):
        """Initialize the ONNX policy.

        Args:
            onnx_path: Path to the ONNX model file.
            device: Device to run inference on ("cuda" or "cpu").
        """
        self.device = device

        # Set up ONNX Runtime session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Load motion_anchor_body_index from ONNX metadata
        model = onnx.load(onnx_path)
        self.motion_anchor_body_index = 0  # default
        for prop in model.metadata_props:
            if prop.key == "motion_anchor_body_index":
                self.motion_anchor_body_index = int(prop.value)
                break

        print(f"[INFO] ONNX model loaded from: {onnx_path}")
        print(f"[INFO] Input names: {self.input_names}")
        print(f"[INFO] Output names: {self.output_names}")
        print(f"[INFO] Motion anchor body index: {self.motion_anchor_body_index}")

    def __call__(self, obs: torch.Tensor, time_step: torch.Tensor, robot_anchor_quat_w: torch.Tensor) -> torch.Tensor:
        """Run inference with the ONNX model (single-step observation, no history).

        Args:
            obs: Observation tensor of shape (num_envs, obs_dim).
            time_step: Time step tensor of shape (num_envs, 1).
            robot_anchor_quat_w: Robot anchor body quaternion in world frame, shape (num_envs, 4).

        Returns:
            Action tensor of shape (num_envs, action_dim).
        """
        # Prepare inputs
        obs_np = obs.cpu().numpy().astype(np.float32)
        time_step_np = time_step.cpu().numpy().astype(np.float32).reshape(-1, 1)  # Ensure shape (num_envs, 1)
        robot_anchor_quat_np = robot_anchor_quat_w.cpu().numpy().astype(np.float32)
        num_envs = obs_np.shape[0]
        obs_dim = obs_np.shape[1]

        # First pass: use dummy obs to get motion data from embedded dataset
        dummy_obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        outputs = self.session.run(
            self.output_names,
            {
                "obs": dummy_obs,
                "time_step": time_step_np,
            },
        )

        # outputs: [actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w]
        # Concatenate joint_pos and joint_vel to form command
        joint_pos = np.asarray(outputs[1])  # shape: (num_envs, 12)
        joint_vel = np.asarray(outputs[2])  # shape: (num_envs, 12)
        command = np.concatenate([joint_pos, joint_vel], axis=-1)  # shape: (num_envs, 24)

        # Get motion anchor quat from body_quat_w using motion_anchor_body_index
        body_quat_w = np.asarray(outputs[4])  # shape: (num_envs, num_bodies, 4)
        motion_anchor_quat_w = body_quat_w[:, self.motion_anchor_body_index, :]  # shape: (num_envs, 4)

        # Compute relative orientation: motion anchor orientation in robot anchor frame
        # ori = quat_multiply(quat_conjugate(robot_anchor_quat), motion_anchor_quat)
        robot_anchor_quat_conj = quat_conjugate(robot_anchor_quat_np)
        relative_ori = quat_multiply(robot_anchor_quat_conj, motion_anchor_quat_w)

        # Convert to rotation matrix and take first 2 columns (6 values)
        mat = matrix_from_quat(relative_ori)  # shape: (num_envs, 3, 3)
        ori_6d = mat[..., :2].reshape(num_envs, -1)  # shape: (num_envs, 6)

        # Replace obs dimensions (single-step, no history)
        # Observation structure: command(24) + motion_anchor_ori_b(6) + base_ang_vel(3) + joint_pos(12) + joint_vel(12) + actions(12) = 69
        obs_modified = obs_np.copy()
        obs_modified[:, :24] = command  # joint_pos + joint_vel
        obs_modified[:, 24:30] = ori_6d  # single orientation (6 dims)

        # DEBUG: Print obs_modified by layer
        debug_obs = {
            "time_step": int(time_step_np[0, 0]),
            "command (0-23)": obs_modified[0, :24],
            "motion_anchor_ori_b (24-29)": obs_modified[0, 24:30],
            "base_ang_vel (30-32)": obs_modified[0, 30:33],
            "joint_pos (33-44)": obs_modified[0, 33:45],
            "joint_vel (45-56)": obs_modified[0, 45:57],
            "actions (57-68)": obs_modified[0, 57:69],
        }
        print("\n[DEBUG] obs_modified (69 dims):")
        pprint(debug_obs)

        # Second pass: run with modified obs to get actions
        outputs = self.session.run(
            self.output_names,
            {
                "obs": obs_modified,
                "time_step": time_step_np,
            },
        )

        # Extract actions (first output)
        actions = torch.from_numpy(outputs[0]).to(self.device)

        return actions

