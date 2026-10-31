
from __future__ import annotations

from gemini_command_interpreter import plan_command_with_gemini


CASES = (
    (
        "coffee mug onto dinner plate",
        "Could you move the coffee mug over to the dinner plate?",
        ("coffee mug", "plate"),
    ),
    (
        "soft drink can onto serving plate",
        "Please set the soft drink can on the serving plate.",
        ("soda can", "plate"),
    ),
    (
        "drinking cup onto dish",
        "Grab the drinking cup and put it onto the dish.",
        ("coffee mug", "plate"),
    ),
    (
        "cola can onto plate",
        "Take the cola can and place it on the plate.",
        ("soda can", "plate"),
    ),
    (
        "pouring liquid is unsupported",
        "Pour the soda from the can into the cup.",
        None,
    ),
    (
        "unknown object is unsupported",
        "Pick up the toaster and place it on the plate.",
        None,
    ),
    (
        "placing beside the plate is unsupported",
        "Move the cup beside the plate.",
        None,
    ),
    (
        "placing above the plate is unsupported",
        "Move the cup above the plate.",
        None,
    ),
    (
        "non-manipulation action is unsupported",
        "Turn on the blender.",
        None,
    ),
)


def main() -> int:
    print(
        f"Running {len(CASES)} cases; at most {len(CASES)} Gemini requests. "
        "Explicit unsupported placement relations may be rejected locally."
    )
    passed = 0
    for label, command, expected in CASES:
        try:
            plan = plan_command_with_gemini(command)
            actual = (plan.object_name, plan.surface_name)
        except ValueError:
            actual = None
        except RuntimeError as error:
            print(f"ERROR during {label}: {error}")
            print(f"Stopped after {passed} passing case(s).")
            return 2

        if actual == expected:
            passed += 1
            print(f"PASS {label}")
        else:
            print(f"FAIL {label}: expected {expected!r}, got {actual!r}")

    print(f"Result: {passed}/{len(CASES)} cases passed.")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
