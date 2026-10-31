
from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom


def convert_asset(
    source: Path,
    destination: Path,
    prim_contains: str,
    scale: float,
    simready_mujoco: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Could not open USD asset: {source}")

    if simready_mujoco:
        root = stage.GetDefaultPrim()
        variant_sets = root.GetVariantSets()
        if "MuJoCo" not in variant_sets.GetNames():
            raise ValueError(f"No MuJoCo variant is available in {source.name}.")
        variant_sets.GetVariantSet("MuJoCo").SetVariantSelection("Enabled")
        for variant_name in ("Newton", "PhysX"):
            if variant_name in variant_sets.GetNames():
                variant_sets.GetVariantSet(variant_name).SetVariantSelection(
                    "Disabled"
                )
        stage.Load()
        prims = Usd.PrimRange(root, Usd.TraverseInstanceProxies())
    else:
        prims = stage.Traverse()

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    xform_cache = UsdGeom.XformCache()
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    filter_text = prim_contains.casefold()

    for prim in prims:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if filter_text not in str(prim.GetPath()).casefold():
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        transform = xform_cache.GetLocalToWorldTransform(prim)
        base = len(vertices)

        for point in points:
            world_point = transform.Transform(point)
            x, y, z = (
                float(world_point[0]) * meters_per_unit,
                float(world_point[1]) * meters_per_unit,
                float(world_point[2]) * meters_per_unit,
            )
            if up_axis == "Y":
                y, z = -z, y
            vertices.append((x * scale, y * scale, z * scale))

        face_offset = 0
        for count in counts:
            polygon = [base + int(i) for i in indices[face_offset : face_offset + count]]
            face_offset += count
            for corner in range(1, len(polygon) - 1):
                triangles.append((polygon[0], polygon[corner], polygon[corner + 1]))

    if not vertices or not triangles:
        raise ValueError(
            f"No triangle mesh matched {prim_contains!r} in {source.name}."
        )

    minimum = tuple(min(vertex[i] for vertex in vertices) for i in range(3))
    maximum = tuple(max(vertex[i] for vertex in vertices) for i in range(3))
    center = tuple((minimum[i] + maximum[i]) / 2 for i in range(3))
    centered_vertices = [
        tuple(vertex[i] - center[i] for i in range(3)) for vertex in vertices
    ]
    dimensions = tuple(maximum[i] - minimum[i] for i in range(3))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii", newline="\n") as obj:
        obj.write(f"# Extracted from {source.name}; prim filter: {prim_contains}\n")
        for vertex in centered_vertices:
            obj.write("v " + " ".join(f"{value:.9g}" for value in vertex) + "\n")
        for triangle in triangles:
            obj.write("f " + " ".join(str(index + 1) for index in triangle) + "\n")

    return center, dimensions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prim-contains", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--simready-mujoco",
        action="store_true",
        help="select the MuJoCo variant and traverse instanced SimReady meshes",
    )
    args = parser.parse_args()

    center, dimensions = convert_asset(
        args.source,
        args.destination,
        args.prim_contains,
        args.scale,
        simready_mujoco=args.simready_mujoco,
    )
    print("OBJ:", args.destination)
    print("Source center (meters):", center)
    print("Exported dimensions (meters):", dimensions)


if __name__ == "__main__":
    main()
