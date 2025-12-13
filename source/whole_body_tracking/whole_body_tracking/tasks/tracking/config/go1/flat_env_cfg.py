from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

from whole_body_tracking.robots.go1 import GO1_ACTION_SCALE, UNITREE_GO1_CFG
from whole_body_tracking.tasks.tracking.config.go1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

import whole_body_tracking.tasks.tracking.mdp as mdp


@configclass
class Go1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Override base_com for Go1 (use trunk instead of torso_link)
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.01, 0.1)},
            },
        )
        self.events.robot_joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            min_step_count_between_reset=720,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "mass_distribution_params": (-1.0, 3.0),
                "operation": "add",
            },
        )


        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = GO1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "trunk"
        self.commands.motion.body_names = [
            "trunk",      # 00
            "FL_hip",     # 01
            "FR_hip",     # 02
            "RL_hip",     # 03
            "RR_hip",     # 04
            "FL_thigh",   # 05
            "FR_thigh",   # 06
            "RL_thigh",   # 07
            "RR_thigh",   # 08
            "FL_calf",    # 09
            "FR_calf",    # 10
            "RL_calf",    # 11
            "RR_calf",    # 12
            "FL_foot",    # 13
            "FR_foot",    # 14
            "RL_foot",    # 15
            "RR_foot",    # 16
        ]
        
        # Override termination conditions for Go1
        self.terminations.anchor_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},
        )
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "FL_foot",
                    "FR_foot",
                    "RL_foot",
                    "RR_foot",
                ],
            },
        )

        self.rewards.dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
        self.rewards.dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
        self.rewards.motion_global_anchor_pos.weight = 1.0
        self.rewards.motion_body_pos.weight = 2.0
@configclass
class Go1FlatWoStateEstimationEnvCfg(Go1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

@configclass
class Go1FlatWoStateEstimationEnvCfg_play(Go1FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.terminations.time_out = None
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "world"  # "world" for free camera, "asset_root" to follow robot

@configclass
class Go1FlatLowFreqEnvCfg(Go1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
