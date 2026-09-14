"""USD material bindings that also apply to instanced Franka finger colliders."""
from typing import Callable
from pxr import UsdShade
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@sim_utils.clone
def spawn_with_contact_material(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    material_path = str(prim.GetPath()) + "/GraspContactMaterial"
    cfg.contact_material.func(material_path, cfg.contact_material)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    for relative_path in cfg.contact_body_paths:
        path = str(prim.GetPath()) + ("/" + relative_path if relative_path else "")
        body = stage.GetPrimAtPath(path)
        if not body.IsValid():
            raise ValueError(f"Cannot bind grasp material: {path}")
        # Bind on the editable body ancestor, so instance-proxy collision meshes inherit it.
        UsdShade.MaterialBindingAPI.Apply(body).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
        )
    return prim


@configclass
class ContactUsdFileCfg(sim_utils.UsdFileCfg):
    func: Callable = spawn_with_contact_material
    contact_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()
    contact_body_paths: tuple[str, ...] = ("",)
