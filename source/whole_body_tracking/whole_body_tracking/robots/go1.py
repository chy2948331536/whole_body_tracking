import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

UNITREE_GO1_CFG = ArticulationCfg(
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path=f"{ASSET_DIR}/chenzheng_go1/go1.usd",
    #     activate_contact_sensors=True,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         retain_accelerations=False,
    #         linear_damping=0.0,
    #         angular_damping=0.0,
    #         max_linear_velocity=1000.0,
    #         max_angular_velocity=1000.0,
    #         max_depenetration_velocity=1.0,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #     ),
    # ),
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        merge_fixed_joints=False,
        asset_path=f"{ASSET_DIR}/chenzheng_go1/go1/urdf/go1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)

GO1_ACTION_SCALE = {}
for a in UNITREE_GO1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            GO1_ACTION_SCALE[n] = 0.25

