
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco


PROJECT_DIR = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_DIR / "scene.xml"
OBJECT_NAMES = ("coffee_mug", "plate", "soda_can")
TARGET_SITE_NAMES = (
    "plate_place_site",
    "table_place_site",
    "table_place_site_2",
    "table_place_site_3",
    "table_place_site_4",
    "table_place_site_5",
)


def named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise KeyError(f"MuJoCo model has no {object_type.name} named {name!r}")
    return object_id


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[float, float, float]:
    body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return tuple(float(value) for value in data.xpos[body_id])


def site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> tuple[float, float, float]:
    site_id = named_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return tuple(float(value) for value in data.site_xpos[site_id])


def print_scene_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name in OBJECT_NAMES:
        print(f"{name}: {body_position(model, data, name)} m")
    for name in TARGET_SITE_NAMES:
        print(f"{name}: {site_position(model, data, name)} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="also open the MuJoCo viewer")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    home_key = named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_objects")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)
    print_scene_state(model, data)

    if args.viewer:
        from mujoco import viewer

        with viewer.launch_passive(model, data) as viewer_handle:
            while viewer_handle.is_running():
                step_start = time.perf_counter()
                mujoco.mj_step(model, data)
                viewer_handle.sync()
                remaining = model.opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    main()
