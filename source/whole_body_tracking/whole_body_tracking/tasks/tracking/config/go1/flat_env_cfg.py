from isaaclab.utils import configclass

from whole_body_tracking.robots.go1 import GO1_ACTION_SCALE, UNITREE_GO1_CFG
from whole_body_tracking.tasks.tracking.config.go1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class Go1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

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


@configclass
class Go1FlatWoStateEstimationEnvCfg(Go1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Go1FlatLowFreqEnvCfg(Go1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
