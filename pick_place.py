
from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

from move_to_coordinate import named_id
from robot_skills import PandaSkills, PickPlaceTargets
from task_planner import (
    OBJECT_REGISTRY,
    SURFACE_REGISTRY,
    TaskPlan,
    format_plan,
    plan_commands,
)


PROJECT_DIR = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_DIR / "scene.xml"


def choose_surface_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan: TaskPlan,
    object_radius: float,
) -> int:
    surface_metadata = SURFACE_REGISTRY[plan.surface_name]
    site_names = surface_metadata.get(
        "candidate_sites", (surface_metadata["site"],)
    )
    destination_body_name = surface_metadata.get("body")
    moving_body_name = OBJECT_REGISTRY[plan.object_name]["body"]
    obstacle_metadata = [
        *OBJECT_REGISTRY.values(),
        *(
            metadata
            for metadata in SURFACE_REGISTRY.values()
            if metadata.get("placement_obstacle", False)
        ),
    ]
    clearance_margin = 0.02

    for site_name in site_names:
        site_id = named_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        candidate_xy = data.site_xpos[site_id, :2]
        blocked = False

        for metadata in obstacle_metadata:
            body_name = metadata.get("body")
            geom_name = metadata.get("geom")
            if (
                not body_name
                or not geom_name
                or body_name in {moving_body_name, destination_body_name}
            ):
                continue

            body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            obstacle_xy = data.xpos[body_id, :2]
            obstacle_radius = float(
                metadata.get("placement_radius", model.geom_size[geom_id, 0])
            )
            required_clearance = (
                object_radius + obstacle_radius + clearance_margin
            )
            if np.linalg.norm(candidate_xy - obstacle_xy) < required_clearance:
                blocked = True
                break

        if not blocked:
            return site_id

    raise ValueError(
        f"No clear placement spot is available on the {plan.surface_name}. "
        "Move an object or choose a different destination."
    )


def _object_is_on_surface(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body_id: int,
    object_geom_id: int,
    surface_geom_id: int,
    height_tolerance: float = 0.012,
    footprint_radius: float | None = None,
) -> bool:
    surface_rotation = data.geom_xmat[surface_geom_id].reshape(3, 3)
    object_rotation = data.geom_xmat[object_geom_id].reshape(3, 3)
    if float(np.dot(surface_rotation[:, 2], object_rotation[:, 2])) < 0.98:
        return False

    surface_delta = surface_rotation.T @ (
        data.xpos[object_body_id] - data.geom_xpos[surface_geom_id]
    )
    object_radius = (
        float(model.geom_size[object_geom_id, 0])
        if footprint_radius is None
        else float(footprint_radius)
    )
    object_half_height = float(model.geom_size[object_geom_id, 1])
    surface_size = model.geom_size[surface_geom_id]
    if model.geom_type[surface_geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        surface_half_height = float(surface_size[2])
        within_footprint = (
            abs(float(surface_delta[0])) <= float(surface_size[0]) - object_radius
            and abs(float(surface_delta[1])) <= float(surface_size[1]) - object_radius
        )
    elif model.geom_type[surface_geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        surface_half_height = float(surface_size[1])
        within_footprint = (
            float(np.linalg.norm(surface_delta[:2]))
            <= float(surface_size[0]) - object_radius
        )
    else:
        return False

    expected_center_height = surface_half_height + object_half_height
    height_error = abs(float(surface_delta[2]) - expected_center_height)
    return within_footprint and height_error <= height_tolerance


def build_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan: TaskPlan,
) -> tuple[PickPlaceTargets, np.ndarray, np.ndarray]:
    object_metadata = OBJECT_REGISTRY[plan.object_name]
    surface_metadata = SURFACE_REGISTRY[plan.surface_name]
    object_body_id = named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, object_metadata["body"]
    )
    object_geom_id = named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, object_metadata["geom"]
    )
    surface_geom_id = named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, surface_metadata["geom"]
    )

    object_center = data.xpos[object_body_id].copy()
    object_radius = float(model.geom_size[object_geom_id, 0])
    placement_radius = float(
        object_metadata.get("placement_radius", object_radius)
    )
    object_width = 2.0 * object_radius
    surface_site_id = choose_surface_site(
        model, data, plan, placement_radius
    )
    surface_position = data.site_xpos[surface_site_id].copy()
    object_half_height = float(model.geom_size[object_geom_id, 1])

                                                                       
    grasp_fraction = float(object_metadata.get("grasp_height_fraction", 0.45))
    plate_grasp_fraction = object_metadata.get("plate_grasp_height_fraction")
    object_is_on_plate = False
    if plate_grasp_fraction is not None:
        plate_geom_id = named_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            SURFACE_REGISTRY["plate"]["geom"],
        )
        object_is_on_plate = _object_is_on_surface(
            model,
            data,
            object_body_id,
            object_geom_id,
            plate_geom_id,
            footprint_radius=placement_radius,
        )
        if object_is_on_plate:
            grasp_fraction = max(grasp_fraction, float(plate_grasp_fraction))
            print(
                "Coffee mug is resting on the plate; using a higher grasp point "
                "to keep the fingers clear of the plate."
            )
    grasp_offset = np.array(
        [0.0, 0.0, object_half_height * grasp_fraction]
    )
    grasp_center = object_center + grasp_offset
    object_approach = grasp_center + np.array([0.0, 0.0, 0.10])
    carry_lift_height = float(object_metadata.get("carry_lift_height", 0.16))
    if object_is_on_plate:
        carry_lift_height = float(
            object_metadata.get("plate_carry_lift_height", carry_lift_height)
        )
        print(
            f"Using a {carry_lift_height:.2f} m lift for the plate re-grasp "
            "to keep the carry pose reachable."
        )
    carry_center = (
        object_center
        + np.array([0.0, 0.0, carry_lift_height])
        + grasp_offset
    )

                                                                              
    place_object_center = surface_position + np.array(
        [0.0, 0.0, object_half_height + 0.002]
    )
    surface_approach_clearance = float(
        object_metadata.get("surface_approach_clearance", 0.10)
    )
    place_center = place_object_center + grasp_offset
    surface_approach = place_center + np.array(
        [0.0, 0.0, surface_approach_clearance]
    )

    targets = PickPlaceTargets(
        object_approach=object_approach,
        object_grasp=grasp_center,
        carry=carry_center,
        surface_approach=surface_approach,
        surface_place=place_center,
        object_body_id=object_body_id,
        initial_object_center=object_center.copy(),
        object_half_height=object_half_height,
        object_radius=object_radius,
        placement_radius=placement_radius,
        surface_site_id=surface_site_id,
        surface_radius=float(model.geom_size[surface_geom_id, 0]),
        grasp_squeeze=float(
            object_metadata.get("grasp_squeeze", 0.0)
        ),
        max_grasp_compression=float(
            object_metadata.get(
                "max_grasp_compression",
                min(0.012, object_width * 0.15),
            )
        ),
        gripper_stiffness_multiplier=float(
            object_metadata.get("gripper_stiffness_multiplier", 1.0)
        ),
    )
    return targets, object_center, surface_position


def object_is_on_destination(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan: TaskPlan,
) -> bool:
    object_metadata = OBJECT_REGISTRY[plan.object_name]
    surface_metadata = SURFACE_REGISTRY[plan.surface_name]
    object_body_id = named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, object_metadata["body"]
    )
    object_geom_id = named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, object_metadata["geom"]
    )
    surface_geom_id = named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, surface_metadata["geom"]
    )

    return _object_is_on_surface(
        model,
        data,
        object_body_id,
        object_geom_id,
        surface_geom_id,
        height_tolerance=0.012,
        footprint_radius=float(
            object_metadata.get(
                "placement_radius", model.geom_size[object_geom_id, 0]
            )
        ),
    )


def run_plan(
    viewer_handle: viewer.Handle,
    plan: TaskPlan,
    skills: PandaSkills,
    targets: PickPlaceTargets,
) -> bool:
    for index, step in enumerate(plan.steps, start=1):
        print(f"Executing step {index}/{len(plan.steps)}: {step.action} {step.target}")
        if not skills.execute_action(viewer_handle, step.action, targets):
            return False
    return True


def resolve_plan(command: str, planner_name: str) -> TaskPlan:
    plans = resolve_plans(command, planner_name)
    if len(plans) != 1:
        raise ValueError(
            "This command contains multiple tasks; use resolve_plans() instead."
        )
    return plans[0]


def resolve_plans(command: str, planner_name: str) -> tuple[TaskPlan, ...]:
    if planner_name == "gemini":
        from gemini_command_interpreter import plan_commands_with_gemini

        return plan_commands_with_gemini(command)
    return plan_commands(command)


def print_task_details(
    command: str,
    planner_name: str,
    plan: TaskPlan,
    object_center: np.ndarray,
    surface_position: np.ndarray,
    task_index: int = 1,
    task_count: int = 1,
) -> None:
    print(f"Command: {command}")
    print(f"Command planner: {planner_name}")
    print(f"Subtask {task_index}/{task_count}:")
    print(format_plan(plan))
    print(
        f"Live {plan.object_name} position:",
        tuple(float(value) for value in object_center),
        "m",
    )
    print(
        f"Live {plan.surface_name} target:",
        tuple(float(value) for value in surface_position),
        "m",
    )


def run_task_sequence(
    viewer_handle: viewer.Handle,
    command: str,
    planner_name: str,
    plans: tuple[TaskPlan, ...],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    skills: PandaSkills,
) -> bool:
    for index, plan in enumerate(plans, start=1):
        if object_is_on_destination(model, data, plan):
            print(
                f"Subtask {index}/{len(plans)} already satisfied: "
                f"{plan.object_name} is already on {plan.surface_name}; skipping it."
            )
            continue

        try:
            targets, object_center, surface_position = build_targets(
                model, data, plan
            )
        except ValueError as error:
            print(f"Could not plan subtask {index}/{len(plans)}: {error}")
            return False

        print_task_details(
            command,
            planner_name,
            plan,
            object_center,
            surface_position,
            task_index=index,
            task_count=len(plans),
        )
        try:
            completed = run_plan(viewer_handle, plan, skills, targets)
            if not completed:
                print(
                    f"Subtask {index}/{len(plans)} did not complete; "
                    "later tasks will not run."
                )
                return False
        except ValueError as error:
            print(f"Motion stopped during subtask {index}/{len(plans)}: {error}")
            return False
        print(f"Subtask {index}/{len(plans)} complete.")
    return True


def _read_interactive_commands(
    commands: queue.Queue[str | None], prompt_ready: threading.Event
) -> None:
    while True:
        prompt_ready.wait()
        prompt_ready.clear()
        try:
            command = input("Task command (or 'cancel'/'quit'): ")
        except EOFError:
            commands.put(None)
            return
        commands.put(command)
        if command.strip().casefold() in {"quit", "exit"}:
            return


def run_interactive_session(
    viewer_handle: viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    skills: PandaSkills,
    planner_name: str,
) -> None:
    commands: queue.Queue[str | None] = queue.Queue()
    prompt_ready = threading.Event()
    threading.Thread(
        target=_read_interactive_commands,
        args=(commands, prompt_ready),
        name="task-command-input",
        daemon=True,
    ).start()

    print(
        "Interactive session ready. Enter one pick-and-place command or task chain at a time; "
        "type 'quit' to close the viewer."
    )
    prompt_ready.set()
    pending_object: str | None = None
    while viewer_handle.is_running():
        try:
            command = commands.get_nowait()
        except queue.Empty:
            if not skills.step_for(viewer_handle, duration=model.opt.timestep):
                return
            continue

        if command is None or command.strip().casefold() in {"quit", "exit"}:
            return
        if not command.strip():
            prompt_ready.set()
            continue
        if command.strip().casefold() in {"cancel", "never mind", "nevermind"}:
            if pending_object is not None:
                print(f"Cancelled the pending request to pick up the {pending_object}.")
                pending_object = None
            else:
                print("There is no incomplete command to cancel.")
            prompt_ready.set()
            continue

        try:
            if planner_name == "gemini":
                from gemini_command_interpreter import interpret_command_with_gemini

                interpretation = interpret_command_with_gemini(
                    command, pending_object=pending_object
                )
                if interpretation.needs_destination:
                    pending_object = interpretation.object_name
                    print(
                        f"I understood that you want to pick up the {pending_object}. "
                        "Where should I place it? You can say 'on the plate' or 'on the table'."
                    )
                    prompt_ready.set()
                    continue
                plans = interpretation.plans
                pending_object = None
            else:
                plans = resolve_plans(command, planner_name)
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"Command not accepted: {error}")
            if pending_object is not None:
                print(
                    f"I am still waiting for a supported destination for the "
                    f"{pending_object}; say 'cancel' to clear it."
                )
            prompt_ready.set()
            continue

        if not plans:
            print("I couldn't build a complete task plan from that command.")
            prompt_ready.set()
            continue

        print(
            f"Accepted {len(plans)} ordered task(s): "
            + "; ".join(
                f"{plan.object_name} to {plan.surface_name}" for plan in plans
            )
        )
        completed = run_task_sequence(
            viewer_handle,
            command,
            planner_name,
            plans,
            model,
            data,
            skills,
        )

        if not completed:
            print(
                "Interactive session stopped after an incomplete task sequence. "
                "The viewer will stay open for inspection; close it to exit."
            )
            while viewer_handle.is_running():
                if not skills.step_for(
                    viewer_handle, duration=model.opt.timestep
                ):
                    break
            return
        print("Command complete. The scene state is retained for the next command.")
        prompt_ready.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="one or more supported pick-and-place tasks in plain text",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="keep one MuJoCo session open and accept sequential commands",
    )
    parser.add_argument(
        "--planner",
        choices=("rules", "gemini"),
        default="rules",
        help="interpret the command with the built-in rules or Gemini API",
    )
    args = parser.parse_args()
    if args.interactive and args.command is not None:
        parser.error("Do not provide a command together with --interactive.")
    command = args.command or "pick up the coffee mug and place it on the plate"
    if not args.interactive:
        try:
            plans = resolve_plans(command, args.planner)
        except (ImportError, RuntimeError, ValueError) as error:
            parser.error(str(error))

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    home_key = named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home_with_objects")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)

    if (
        not args.interactive
        and len(plans) == 1
        and object_is_on_destination(model, data, plans[0])
    ):
        plan = plans[0]
        print(
            f"Task already satisfied: {plan.object_name} is already on "
            f"the {plan.surface_name}; no motion is needed."
        )
        return

    skills = PandaSkills(model, data)

    with viewer.launch_passive(model, data) as viewer_handle:
        if args.interactive:
            run_interactive_session(
                viewer_handle, model, data, skills, args.planner
            )
            return

        print(
            f"Validated {len(plans)} ordered task(s): "
            + "; ".join(
                f"{plan.object_name} to {plan.surface_name}" for plan in plans
            )
        )
        print("Starting scripted task sequence; close the viewer to exit.")
        sequence_complete = run_task_sequence(
            viewer_handle,
            command,
            args.planner,
            plans,
            model,
            data,
            skills,
        )

        if sequence_complete:
            print("Sequence complete. The viewer will stay open until you close it.")
        else:
            print("The sequence stopped early. The viewer will stay open for inspection.")
        while viewer_handle.is_running():
            if not skills.step_for(viewer_handle, duration=model.opt.timestep):
                break


if __name__ == "__main__":
    main()
