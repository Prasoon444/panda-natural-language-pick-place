
from __future__ import annotations

from dataclasses import dataclass
import re


OBJECT_REGISTRY = {
    "coffee mug": {
        "aliases": (
            "coffee mug",
            "coffee mug with a handle",
            "mug",
            "mug with a handle",
            "handled mug",
            "cup",
            "coffee cup",
            "drinking cup",
        ),
        "body": "coffee_mug",
        "geom": "coffee_mug_geom",
        "grasp_squeeze": 0.012,
        "max_grasp_compression": 0.012,
        "gripper_stiffness_multiplier": 4.0,
                                                                           
        "placement_radius": 0.0672,
                                                                         
                                                                 
        "plate_grasp_height_fraction": 0.78,
                                                                            
                                                                          
        "plate_carry_lift_height": 0.08,
    },
    "soda can": {
        "aliases": (
            "soda can",
            "soda_can",
            "sodacan",
            "can",
            "beverage can",
            "soft drink can",
            "drink can",
            "cola can",
        ),
        "body": "soda_can",
        "geom": "soda_can_geom",
                                                                            
                                                                   
        "grasp_squeeze": 0.012,
        "max_grasp_compression": 0.015,
        "gripper_stiffness_multiplier": 4.0,
    },
}

SURFACE_REGISTRY = {
    "plate": {
        "aliases": ("plate", "dish", "dinner plate", "serving plate"),
        "body": "plate",
        "placement_obstacle": True,
        "site": "plate_place_site",
        "geom": "plate_geom",
    },
    "table": {
        "aliases": ("table", "tabletop", "table top", "table surface"),
        "body": "table",
        "site": "table_place_site",
        "candidate_sites": (
            "table_place_site",
            "table_place_site_2",
            "table_place_site_3",
            "table_place_site_4",
            "table_place_site_5",
        ),
        "geom": "table_top",
    },
}


@dataclass(frozen=True)
class TaskStep:
    action: str
    target: str


@dataclass(frozen=True)
class TaskPlan:
    command: str
    object_name: str
    surface_name: str
    steps: tuple[TaskStep, ...]


_COMMAND_PATTERN = re.compile(
    r"^(?:please\s+)?(?:pick\s+up|pick|grab)\s+(?:the\s+)?"
    r"(?P<object>[\w-]+(?:\s+[\w-]+)*)"
    r"\s+and(?:\s+then)?\s+(?:place|put|set)"
    r"(?:\s+(?:it|(?:the\s+)?[\w-]+(?:\s+[\w-]+)*))?"
    r"\s+(?:on|onto)\s+(?:the\s+)?(?P<surface>[\w-]+(?:\s+[\w-]+)*)$",
    re.IGNORECASE,
)
_TASK_START_PATTERN = r"(?:please\s+)?(?:pick\s+up|pick|grab)\b"
_SEQUENCE_SEPARATOR = re.compile(
    r"(?:[.!?]\s*(?:then\s+)?|;\s*(?:then\s+)?|,\s*(?:then\s+)?|"
    r"\s+(?:and\s+then|then|and)\s+)"
    rf"(?={_TASK_START_PATTERN})",
    re.IGNORECASE,
)
MAX_TASKS_PER_COMMAND = 5


def _alias_lookup(registry: dict[str, dict], noun: str, category: str) -> str:
    normalized = " ".join(noun.casefold().split())
    for canonical_name, metadata in registry.items():
        if normalized in metadata["aliases"]:
            return canonical_name
    supported = ", ".join(sorted(registry))
    raise ValueError(
        f"I don't know that {category} ({noun!r}). Currently supported: {supported}."
    )


def plan_from_entities(
    command: str, object_phrase: str, surface_phrase: str
) -> TaskPlan:
    object_name = _alias_lookup(
        OBJECT_REGISTRY, object_phrase, "object"
    )
    surface_name = _alias_lookup(
        SURFACE_REGISTRY, surface_phrase, "placement surface"
    )
    if object_name == surface_name:
        raise ValueError("The object and placement surface must be different.")

    steps = (
        TaskStep("approach", object_name),
        TaskStep("grasp", object_name),
        TaskStep("lift", object_name),
        TaskStep("move above", surface_name),
        TaskStep("lower onto", surface_name),
        TaskStep("release", object_name),
        TaskStep("retreat", surface_name),
    )
    return TaskPlan(
        command=command,
        object_name=object_name,
        surface_name=surface_name,
        steps=steps,
    )


def plan_command(command: str) -> TaskPlan:
    plans = plan_commands(command)
    if len(plans) != 1:
        raise ValueError(
            "This command contains multiple tasks. Use plan_commands() to "
            "resolve a task sequence."
        )
    return plans[0]


def plan_commands(command: str) -> tuple[TaskPlan, ...]:
    normalized = " ".join(command.strip().rstrip(".!?").split())
    segments = [
        segment.strip()
        for segment in _SEQUENCE_SEPARATOR.split(normalized)
        if segment.strip()
    ]
    if not segments or len(segments) > MAX_TASKS_PER_COMMAND:
        raise ValueError(
            f"A command must contain between 1 and {MAX_TASKS_PER_COMMAND} "
            "pick-and-place tasks."
        )

    plans: list[TaskPlan] = []
    for segment in segments:
        match = _COMMAND_PATTERN.fullmatch(segment)
        if match is None:
            raise ValueError(
                "Use a command like 'pick up the coffee mug and place it on the plate'. "
                "For multiple tasks, connect complete commands with 'then', "
                "'and then', commas, or semicolons."
            )
        plans.append(
            plan_from_entities(
                segment,
                match.group("object"),
                match.group("surface"),
            )
        )
    return tuple(plans)


def format_plan(plan: TaskPlan) -> str:
    lines = [
        f"Resolved task: pick {plan.object_name} and place it on {plan.surface_name}.",
        "Action plan:",
    ]
    lines.extend(
        f"  {index}. {step.action} {step.target}"
        for index, step in enumerate(plan.steps, start=1)
    )
    return "\n".join(lines)
