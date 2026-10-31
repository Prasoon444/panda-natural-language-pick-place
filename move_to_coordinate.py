
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_DIR / "scene.xml"
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
TOOL_SITE_NAME = "tool_center"
GRIPPER_ACTUATOR_NAME = "actuator8"
ROTATION_WEIGHT = 0.12


def named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise KeyError(f"MuJoCo model has no {object_type.name} named {name!r}")
    return object_id


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    return 0.5 * np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=float,
    )


def solve_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    max_iterations: int = 300,
    orientation_tolerance: float = 0.04,
) -> np.ndarray:
    joint_ids = np.array(
        [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES],
        dtype=int,
    )
    qpos_addresses = model.jnt_qposadr[joint_ids].copy()
    dof_addresses = model.jnt_dofadr[joint_ids].copy()
    initial_qpos = data.qpos.copy()
    lower = model.jnt_range[joint_ids, 0] + 1e-4
    upper = model.jnt_range[joint_ids, 1] - 1e-4
    damping = 0.04

    try:
        for _ in range(max_iterations):
            mujoco.mj_forward(model, data)
            current_position = data.site_xpos[site_id].copy()
            current_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
            position_error = target_position - current_position
            orientation_error = _rotation_error(target_rotation, current_rotation)

            if (
                np.linalg.norm(position_error) < 0.004
                and np.linalg.norm(orientation_error) < orientation_tolerance
            ):
                return data.qpos[qpos_addresses].copy()

            jacobian_position = np.zeros((3, model.nv), dtype=float)
            jacobian_rotation = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(
                model, data, jacobian_position, jacobian_rotation, site_id
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, dof_addresses],
                    ROTATION_WEIGHT * jacobian_rotation[:, dof_addresses],
                )
            )
            error = np.concatenate(
                (position_error, ROTATION_WEIGHT * orientation_error)
            )
            system = jacobian @ jacobian.T + damping**2 * np.eye(6)
            delta_q = jacobian.T @ np.linalg.solve(system, error)
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > 0.08:
                delta_q *= 0.08 / delta_norm
            data.qpos[qpos_addresses] = np.clip(
                data.qpos[qpos_addresses] + delta_q, lower, upper
            )

        mujoco.mj_forward(model, data)
        final_error = np.linalg.norm(target_position - data.site_xpos[site_id])
        final_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
        final_orientation_error = np.linalg.norm(
            _rotation_error(target_rotation, final_rotation)
        )
        raise ValueError(
            f"IK did not converge to the requested XYZ target; final position error is {final_error:.3f} m. "
            f"Final orientation error is {final_orientation_error:.3f} rad. "
            "The pose may be outside the Panda's reachable workspace."
        )
    finally:
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, help="target world X coordinate in meters")
    parser.add_argument("y", type=float, help="target world Y coordinate in meters")
    parser.add_argument("z", type=float, help="target world Z coordinate in meters")
    parser.add_argument("--gripper", choices=("open", "close"), default="open")
    parser.add_argument("--move-seconds", type=float, default=3.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    home_key = named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_objects")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)

    site_id = named_id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME)
    target_position = np.array([args.x, args.y, args.z], dtype=float)
    target_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    target_arm_qpos = solve_position_ik(
        model, data, site_id, target_position, target_rotation
    )

    arm_actuator_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ARM_ACTUATOR_NAMES
    ]
    gripper_actuator_id = named_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
    )
    for actuator_id, qpos in zip(arm_actuator_ids, target_arm_qpos):
        data.ctrl[actuator_id] = qpos

                                                                                
    data.ctrl[gripper_actuator_id] = 255
    move_steps = max(1, int(args.move_seconds / model.opt.timestep))

    from mujoco import viewer

    with viewer.launch_passive(model, data) as viewer_handle:
        step = 0
        while viewer_handle.is_running():
            if step == move_steps:
                data.ctrl[gripper_actuator_id] = (
                    255 if args.gripper == "open" else 0
                )
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer_handle.sync()
            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
            step += 1

    mujoco.mj_forward(model, data)
    actual_position = data.site_xpos[site_id]
    print(
        "tool_center:",
        tuple(float(value) for value in actual_position),
        "m",
    )


if __name__ == "__main__":
    main()
