
from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco
import numpy as np
from mujoco import viewer

from move_to_coordinate import (
    ARM_ACTUATOR_NAMES,
    GRIPPER_ACTUATOR_NAME,
    TOOL_SITE_NAME,
    named_id,
    solve_position_ik,
)


@dataclass(frozen=True)
class PickPlaceTargets:

    object_approach: np.ndarray
    object_grasp: np.ndarray
    carry: np.ndarray
    surface_approach: np.ndarray
    surface_place: np.ndarray
    object_body_id: int
    initial_object_center: np.ndarray
    object_half_height: float
    object_radius: float
    placement_radius: float
    surface_site_id: int
    surface_radius: float
    grasp_squeeze: float
    max_grasp_compression: float
    gripper_stiffness_multiplier: float


class PandaSkills:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.tool_site_id = named_id(
            model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME
        )
        self.finger_body_ids = {
            named_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            named_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        }
        self.finger_body_names = {
            named_id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"): "left",
            named_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"): "right",
        }
        self.arm_actuator_ids = np.array(
            [
                named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ],
            dtype=int,
        )
        self.gripper_actuator_id = named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        finger_joint_id = named_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"
        )
        self.max_gripper_width = 2.0 * float(
            model.jnt_range[finger_joint_id, 1]
        )
        self.gripper_ctrl_range = model.actuator_ctrlrange[
            self.gripper_actuator_id
        ].copy()
        self._base_gripper_gain = float(
            model.actuator_gainprm[self.gripper_actuator_id, 0]
        )
        self._base_gripper_position_bias = float(
            model.actuator_biasprm[self.gripper_actuator_id, 1]
        )
        joint_ids = np.array(
            [
                named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                for i in range(1, 8)
            ],
            dtype=int,
        )
        self.arm_qpos_addresses = model.jnt_qposadr[joint_ids].copy()
        self.tool_rotation = data.site_xmat[self.tool_site_id].reshape(3, 3).copy()

    def _set_gripper_stiffness(self, multiplier: float) -> None:
        multiplier = max(1.0, float(multiplier))
        self.model.actuator_gainprm[self.gripper_actuator_id, 0] = (
            self._base_gripper_gain * multiplier
        )
        self.model.actuator_biasprm[self.gripper_actuator_id, 1] = (
            self._base_gripper_position_bias * multiplier
        )

    def _restore_gripper_stiffness(self) -> None:
        self._set_gripper_stiffness(1.0)

    def step_for(
        self, viewer_handle: viewer.Handle, duration: float
    ) -> bool:
        num_steps = max(1, int(duration / self.model.opt.timestep))
        for _ in range(num_steps):
            if not viewer_handle.is_running():
                return False
            step_start = time.perf_counter()
            mujoco.mj_step(self.model, self.data)
            viewer_handle.sync()
            remaining = self.model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
        return True

    def move_tool_to(
        self,
        viewer_handle: viewer.Handle,
        target_position: np.ndarray,
        duration: float,
        orientation_tolerance: float = 0.04,
    ) -> bool:
        target_qpos = solve_position_ik(
            self.model,
            self.data,
            self.tool_site_id,
            target_position,
            self.tool_rotation,
            orientation_tolerance=orientation_tolerance,
        )
        start_qpos = self.data.qpos[self.arm_qpos_addresses].copy()
        num_steps = max(1, int(duration / self.model.opt.timestep))

        for step in range(1, num_steps + 1):
            if not viewer_handle.is_running():
                return False
            fraction = step / num_steps
            self.data.ctrl[self.arm_actuator_ids] = start_qpos + fraction * (
                target_qpos - start_qpos
            )
            step_start = time.perf_counter()
            mujoco.mj_step(self.model, self.data)
            viewer_handle.sync()
            remaining = self.model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
        return True

    def set_gripper(
        self, viewer_handle: viewer.Handle, opened: bool, duration: float = 0.7
    ) -> bool:
        target_width = self.max_gripper_width if opened else 0.0
        return self.set_gripper_aperture(
            viewer_handle, target_width, duration=duration
        )

    def set_gripper_aperture(
        self,
        viewer_handle: viewer.Handle,
        aperture: float,
        duration: float = 0.7,
    ) -> bool:
        target_width = float(
            np.clip(aperture, 0.0, self.max_gripper_width)
        )
        fraction = target_width / self.max_gripper_width
        ctrl_min, ctrl_max = self.gripper_ctrl_range
        self.data.ctrl[self.gripper_actuator_id] = ctrl_min + fraction * (
            ctrl_max - ctrl_min
        )
        return self.step_for(viewer_handle, duration)

    def contacting_fingers(self, object_body_id: int) -> set[str]:
        contacting: set[str] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if body1 == object_body_id and body2 in self.finger_body_ids:
                contacting.add(self.finger_body_names[body2])
            elif body2 == object_body_id and body1 in self.finger_body_ids:
                contacting.add(self.finger_body_names[body1])
        return contacting

    def verify_placement(self, targets: PickPlaceTargets) -> tuple[bool, float, float]:
        actual = self.data.xpos[targets.object_body_id]
        surface = self.data.site_xpos[targets.surface_site_id]
        xy_error = float(np.linalg.norm(actual[:2] - surface[:2]))
        expected_z = surface[2] + targets.object_half_height
        z_error = float(abs(actual[2] - expected_z))
        xy_tolerance = min(
            0.05,
            max(0.0, targets.surface_radius - targets.placement_radius),
        )
        return xy_error <= xy_tolerance and z_error <= 0.025, xy_error, z_error

    def approach_object(
        self, viewer_handle: viewer.Handle, target: np.ndarray
    ) -> bool:
        if not self.set_gripper(viewer_handle, opened=True):
            return False

        clearance_target = target.copy()
        clearance_target[2] += 0.02
        return self.move_tool_to(
            viewer_handle, clearance_target, duration=2.5
        ) and self.move_tool_to(viewer_handle, target, duration=0.8)

    def grasp_object(
        self, viewer_handle: viewer.Handle, targets: PickPlaceTargets
    ) -> bool:
        if not self.move_tool_to(
            viewer_handle,
            targets.object_grasp,
            duration=1.5,
        ):
            return False

        self._set_gripper_stiffness(targets.gripper_stiffness_multiplier)
        object_width = 2.0 * targets.object_radius
        max_compression = targets.max_grasp_compression
        minimum_width = max(0.0, object_width - max_compression)
        target_grasp_width = max(
            minimum_width, object_width - targets.grasp_squeeze
        )
        grasp_width = min(self.max_gripper_width, object_width + 0.004)
        aperture_step = 0.002 if targets.grasp_squeeze > 0.0 else 0.0005
        print(
            f"Closing gripper adaptively for the object's "
            f"{object_width:.3f} m collision width; maximum squeeze is "
            f"{max_compression:.3f} m; target squeeze is "
            f"{targets.grasp_squeeze:.3f} m; position stiffness is "
            f"{targets.gripper_stiffness_multiplier:.1f}x."
        )

        while True:
            if not self.set_gripper_aperture(
                viewer_handle, grasp_width, duration=0.12
            ):
                return False
            contacting = self.contacting_fingers(targets.object_body_id)
            if (
                contacting == {"left", "right"}
                and grasp_width <= target_grasp_width + 1e-9
            ):
                print(
                    "Grasp check passed: both fingers contact the object at "
                    f"a {grasp_width:.3f} m aperture with "
                    f"{object_width - grasp_width:.3f} m squeeze."
                )
                return True
            if grasp_width <= minimum_width:
                break
            grasp_width = max(minimum_width, grasp_width - aperture_step)

        print(
            "Grasp check failed at the object-specific squeeze limit: contact "
            "detected on "
            f"{', '.join(sorted(contacting)) if contacting else 'neither finger'}. "
            "Opening the gripper and returning to the object approach pose."
        )
        self._restore_gripper_stiffness()
        self.set_gripper(viewer_handle, opened=True, duration=0.4)
        self.move_tool_to(viewer_handle, targets.object_approach, duration=1.2)
        return False

    def lift_object(
        self, viewer_handle: viewer.Handle, targets: PickPlaceTargets
    ) -> bool:
        if not self.move_tool_to(
            viewer_handle,
            targets.carry,
            duration=1.5,
        ):
            return False
        actual = self.data.xpos[targets.object_body_id]
        height_gain = float(actual[2] - targets.initial_object_center[2])
        horizontal_drift = float(
            np.linalg.norm(actual[:2] - targets.initial_object_center[:2])
        )
        expected_lift = float(targets.carry[2] - targets.object_grasp[2])
        minimum_height_gain = max(0.04, expected_lift - 0.025)
        if height_gain >= minimum_height_gain and horizontal_drift <= 0.06:
            print(
                f"Lift check passed: object rose {height_gain:.3f} m "
                f"with {horizontal_drift:.3f} m horizontal drift."
            )
            return True

        print(
            "Lift check failed: "
            f"height gain={height_gain:.3f} m, "
            f"horizontal drift={horizontal_drift:.3f} m. "
            f"Expected at least {minimum_height_gain:.3f} m."
        )
        if self.contacting_fingers(targets.object_body_id) == {"left", "right"}:
            tool_to_object_offset = (
                self.data.site_xpos[self.tool_site_id]
                - self.data.xpos[targets.object_body_id]
            )
            set_down_target = (
                targets.initial_object_center + tool_to_object_offset
            )
            print("The grasp still has both contacts; lowering the object back onto the table.")
            try:
                self.move_tool_to(viewer_handle, set_down_target, duration=1.5)
            except ValueError as error:
                print(f"Could not safely lower the object: {error}")

        self._restore_gripper_stiffness()
        self.set_gripper(viewer_handle, opened=True, duration=0.5)
        self.move_tool_to(viewer_handle, targets.object_approach, duration=1.5)
        return False

    def move_above_surface(
        self, viewer_handle: viewer.Handle, targets: PickPlaceTargets
    ) -> bool:
        return self.move_tool_to(
            viewer_handle,
            targets.surface_approach,
            duration=2.5,
        )

    def lower_onto_surface(
        self, viewer_handle: viewer.Handle, targets: PickPlaceTargets
    ) -> bool:
        return self.move_tool_to(
            viewer_handle,
            targets.surface_place,
            duration=1.5,
        )

    def release_object(
        self, viewer_handle: viewer.Handle, targets: PickPlaceTargets
    ) -> bool:
        self._restore_gripper_stiffness()
        if not self.set_gripper(
            viewer_handle, opened=True, duration=0.8
        ) or not self.step_for(viewer_handle, duration=0.5):
            return False

        passed, xy_error, z_error = self.verify_placement(targets)
        if passed:
            print(
                "Placement check passed: "
                f"XY error={xy_error:.3f} m, vertical error={z_error:.3f} m."
            )
            return True

        print(
            "Placement check failed: "
            f"XY error={xy_error:.3f} m, vertical error={z_error:.3f} m. "
            "Stopping the plan and retreating to a safe pose."
        )
        self.move_tool_to(viewer_handle, targets.surface_approach, duration=1.5)
        return False

    def retreat_from_surface(
        self, viewer_handle: viewer.Handle, targets: PickPlaceTargets
    ) -> bool:
        start = self.data.site_xpos[self.tool_site_id].copy()
        target = targets.surface_approach.copy()
                                                                         
                                                                     
        target[2] += 0.03
        distance = float(np.linalg.norm(target - start))
        segment_count = max(1, int(np.ceil(distance / 0.02)))
        segment_duration = 1.5 / segment_count

                                                                              
                                                                              
        relaxed_orientation = False
        for index in range(1, segment_count + 1):
            fraction = index / segment_count
            waypoint = start + fraction * (target - start)
                                                                          
                                                                          
                                                                             
                                               
            orientation_tolerance = 0.04 if fraction <= 0.4 else 0.10
            if orientation_tolerance > 0.04 and not relaxed_orientation:
                print(
                    "Retreat cleared the surface; allowing a small tool-angle "
                    "tolerance for the remaining waypoints."
                )
                relaxed_orientation = True
            if not self.move_tool_to(
                viewer_handle,
                waypoint,
                duration=segment_duration,
                orientation_tolerance=orientation_tolerance,
            ):
                return False

        if not self.step_for(viewer_handle, duration=0.5):
            return False

        passed, xy_error, z_error = self.verify_placement(targets)
        if not passed:
            print(
                "Post-retreat placement check failed: "
                f"XY error={xy_error:.3f} m, vertical error={z_error:.3f} m. "
                "The object moved during retreat; stopping before the next task."
            )
            return False
        print(
            "Post-retreat placement check passed: "
            f"XY error={xy_error:.3f} m, vertical error={z_error:.3f} m."
        )
        return True

    def execute_action(
        self,
        viewer_handle: viewer.Handle,
        action: str,
        targets: PickPlaceTargets,
    ) -> bool:
        skills = {
            "approach": lambda: self.approach_object(
                viewer_handle, targets.object_approach
            ),
            "grasp": lambda: self.grasp_object(viewer_handle, targets),
            "lift": lambda: self.lift_object(viewer_handle, targets),
            "move above": lambda: self.move_above_surface(
                viewer_handle, targets
            ),
            "lower onto": lambda: self.lower_onto_surface(
                viewer_handle, targets
            ),
            "release": lambda: self.release_object(viewer_handle, targets),
            "retreat": lambda: self.retreat_from_surface(
                viewer_handle, targets
            ),
        }
        try:
            skill = skills[action]
        except KeyError as error:
            raise ValueError(f"No Panda skill is registered for action {action!r}.") from error
        return skill()
