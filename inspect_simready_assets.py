
from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", type=Path)
    args = parser.parse_args()

    for asset_path in args.assets:
        stage = Usd.Stage.Open(str(asset_path))
        if stage is None:
            raise RuntimeError(f"Could not open USD asset: {asset_path}")

        print(f"\n{asset_path.name}")
        print("  meters_per_unit:", UsdGeom.GetStageMetersPerUnit(stage))
        print("  up_axis:", UsdGeom.GetStageUpAxis(stage))
        xform_cache = UsdGeom.XformCache()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get() or []
                counts = mesh.GetFaceVertexCountsAttr().Get() or []
                transform = xform_cache.GetLocalToWorldTransform(prim)
                world_points = [transform.Transform(point) for point in points]
                if world_points:
                    minimum = tuple(
                        min(point[axis] for point in world_points)
                        for axis in range(3)
                    )
                    maximum = tuple(
                        max(point[axis] for point in world_points)
                        for axis in range(3)
                    )
                    dimensions = tuple(
                        maximum[axis] - minimum[axis] for axis in range(3)
                    )
                    bounds_text = (
                        f"bounds {tuple(round(v, 4) for v in minimum)} to "
                        f"{tuple(round(v, 4) for v in maximum)} m; "
                        f"size {tuple(round(v, 4) for v in dimensions)} m"
                    )
                else:
                    bounds_text = "no vertices"
                print(
                    f"  mesh {prim.GetPath()}: "
                    f"{len(points)} vertices, {len(counts)} faces; {bounds_text}"
                )
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                mass = mass_api.GetMassAttr().Get()
                if mass is not None:
                    print(f"  mass {prim.GetPath()}: {mass}")


if __name__ == "__main__":
    main()
