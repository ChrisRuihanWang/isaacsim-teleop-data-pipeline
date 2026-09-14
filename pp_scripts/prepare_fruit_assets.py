"""Build centered, metre-scale rigid fruit assets from the downloaded originals."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1] / "assets/fruits"


def main():
    for entry in json.loads((ROOT.parent / "task_assets.json").read_text())["files"]:
        p = Path(__file__).resolve().parents[1] / entry["path"]
        algorithm = "sha256" if "sha256" in entry else "md5"
        if hashlib.new(algorithm, p.read_bytes()).hexdigest() != entry[algorithm]:
            raise ValueError(f"Asset checksum mismatch: {p}")
    metadata = {}
    for name, filename, width, mass in (
        ("banana", "011_banana.usd", None, 0.12),
        ("apple", "food_apple_01_1k.usdc", 0.065, 0.10),
    ):
        folder = ROOT / name
        original = Usd.Stage.Open(str(folder / filename))
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        bounds = bbox.ComputeWorldBound(original.GetPseudoRoot()).ComputeAlignedRange()
        size, center = bounds.GetSize(), bounds.GetMidpoint()
        scale = 1.0 if width is None else width / max(size[0], size[1])
        out = folder / "asset.usda"
        stage = Usd.Stage.CreateNew(str(out))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Fruit").GetPrim()
        stage.SetDefaultPrim(root)
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        visual = UsdGeom.Xform.Define(stage, "/Fruit/Visual")
        model = stage.DefinePrim("/Fruit/Visual/Model")
        model.GetReferences().AddReference(filename)
        # Parent transform centers and scales the original root without changing its children.
        visual.AddTranslateOp().Set(-center * scale)
        visual.AddScaleOp().Set(Gf.Vec3f(scale))
        # Original assets may contain authored root transforms; preserve them inside a child.
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition" if name == "banana" else "convexHull")
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                value = attr.Get()
                if not value or not value.path or value.path == "OmniPBR.mdl":
                    continue
                if not value.resolvedPath:
                    candidate = folder / "textures" / Path(value.path).name
                    if not candidate.exists() and candidate.suffix == ".exr":
                        candidate = candidate.with_suffix(".jpg")
                    if not candidate.exists():
                        raise FileNotFoundError(value.path)
                    attr.Set(Sdf.AssetPath(candidate.relative_to(folder).as_posix()))
        stage.GetRootLayer().Save()
        vertices = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                vertices.extend([list(transform.Transform(Gf.Vec3d(*map(float, point))))
                                 for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
        vertices = np.asarray(vertices)
        hull = vertices[ConvexHull(vertices).vertices]
        # Mid-length banana flesh is offset from the bounding-box center by its curve.
        section = vertices[np.abs(vertices[:, 0]) < 0.005]
        anchor = (section.min(axis=0) + section.max(axis=0)) / 2 if name == "banana" else np.zeros(3)
        anchor[0] = 0.0
        metadata[name] = {"grasp_anchor_local_m": anchor.tolist(), "hull_vertices_m": hull.tolist(),"size_m": list(size * scale), "half_extents_m": list(size * scale / 2),
                          "scale": scale, "mass_kg": mass, "collision": "convexDecomposition" if name == "banana" else "convexHull"}
    (ROOT / "geometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print({name: {"size_m": data["size_m"], "hull_points": len(data["hull_vertices_m"])} for name, data in metadata.items()})


if __name__ == "__main__":
    main()
