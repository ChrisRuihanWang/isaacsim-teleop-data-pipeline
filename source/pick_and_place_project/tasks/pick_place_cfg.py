from pathlib import Path
from dataclasses import fields
import json

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.managers import SceneEntityCfg
from pick_and_place_project.tasks.mdp.actions import FrankaGripperActionCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from pick_and_place_project.tasks.mdp.success import placed_on_target
from pick_and_place_project.tasks.mdp.contact import ContactUsdFileCfg

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
for relative_path in ("fruits/geometry.json", "fruits/banana/asset.usda", "fruits/apple/asset.usda",
                      "containers/nvidia_klt/small_KLT.usd"):
    if not (ASSET_ROOT / relative_path).is_file():
        raise FileNotFoundError(
            f"Missing task asset: {relative_path}. Run python pp_scripts/fetch_task_assets.py "
            "with the Isaac Sim Python environment before launching."
        )
FRUIT_GEOMETRY = json.loads((ASSET_ROOT / "fruits/geometry.json").read_text())


from pick_and_place_project.tasks.mdp import observation as my_obs


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)
    success: DoneTerm = DoneTerm(func=placed_on_target)


@configclass
class ActionsCfg:
    arm_action: DifferentialInverseKinematicsActionCfg | None = None
    gripper_action: FrankaGripperActionCfg | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # robot arm joints
        joint_pos_rel: ObsTerm = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel: ObsTerm = ObsTerm(func=mdp.joint_vel_rel)

        # gripper joints (2 dims)
        gripper_joint_pos_rel: ObsTerm = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("panda_finger_joint1", "panda_finger_joint2"),
                )
            },
        )
        gripper_joint_vel_rel: ObsTerm = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("panda_finger_joint1", "panda_finger_joint2"),
                )
            },
        )


        gripper_width: ObsTerm = ObsTerm(
            func=my_obs.franka_gripper_width,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("panda_finger_joint1", "panda_finger_joint2"),
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImagesCfg(ObsGroup):

        rgb: ObsTerm = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        wrist_rgb: ObsTerm = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        oblique_rgb: ObsTerm = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("oblique_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    images: ImagesCfg = ImagesCfg()


@configclass
class SceneCfg(InteractiveSceneCfg):
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    decimation: int = 4
    grasp_offset_z: float = 0.097
    finger_static_friction: float = 1.2
    finger_dynamic_friction: float = 1.0
    fruit_static_friction: float = 0.8
    fruit_dynamic_friction: float = 0.6
    scene: SceneCfg = SceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.commands = None
        self.rewards = None

        # ground
        self.scene.ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )


        self.scene.dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(
                intensity=1000.0,
                color=(1.0, 1.0, 1.0),
            ),
        )

        # robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Apply finger-only material without modifying the shared upstream robot asset.
        robot_spawn = self.scene.robot.spawn
        self.scene.robot.spawn = ContactUsdFileCfg(
            **{field.name: getattr(robot_spawn, field.name) for field in fields(robot_spawn) if field.name != "func"},
            contact_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.finger_static_friction, dynamic_friction=self.finger_dynamic_friction,
                restitution=0.0, friction_combine_mode="average",
            ),
            contact_body_paths=("panda_leftfinger", "panda_rightfinger"),
        )
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 12
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        self.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002, rest_offset=0.0
        )

        # Local textured fruit assets contain centered geometry and convex collision meshes.
        for name, xy in (("banana", (0.45, -0.18)), ("apple", (0.50, 0.17))):
            geometry = FRUIT_GEOMETRY[name]
            setattr(self.scene, name, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=ContactUsdFileCfg(
                    usd_path=str(ASSET_ROOT / "fruits" / name / "asset.usda"),
                    contact_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=self.fruit_static_friction, dynamic_friction=self.fruit_dynamic_friction,
                        restitution=0.0, friction_combine_mode="average",
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=4,
                        angular_damping=0.5,
                        max_depenetration_velocity=1.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(*xy, geometry["half_extents_m"][2] + 0.005),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ))

        # Retain the asset name basket for task code; the KLT has an open cavity.
        self.scene.basket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ASSET_ROOT / "containers/nvidia_klt/small_KLT.usd"),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.75, 0.0, 0.074), rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # actions: IK + gripper
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, self.grasp_offset_z],
            ),
        )

        self.actions.gripper_action = FrankaGripperActionCfg(
            asset_name="robot",
            joint_names=("panda_finger_joint1", "panda_finger_joint2"),
            open_pos=0.04,
            close_pos=0.0,
        )


        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/OverheadCamera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.6, 0.0, 1.2),
                rot=(0.0, 1.0, 0.0, 0.0),
                convention="ros",
            ),
        )


        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.00, 0.08),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
        )

        self.scene.oblique_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ObliqueCamera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.05, 0.8, 0.65),
                rot=(-0.12057844, 0.22287456, 0.85082512, -0.46030901),
                convention="ros",
            ),
        )



@configclass
class PickPlaceEnvCfg_PLAY(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
