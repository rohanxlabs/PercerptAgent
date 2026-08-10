"""Safe, explicit simulation adapter for actuator commands.

This module intentionally does not control physical hardware.  Replace the adapter
with a reviewed hardware-specific implementation before enabling a real robot.
"""

from dataclasses import dataclass, field
from time import time


@dataclass
class SimulatedActuator:
    """Records validated actions so the full agent-to-action path is observable."""

    actions: list[dict] = field(default_factory=list)

    def execute(self, target: str, action: str, parameters: dict) -> dict:
        if target not in {"robot_arm", "camera_ptz", "alarm"}:
            raise ValueError(f"Unsupported simulated target: {target}")
        if not action or len(action) > 64:
            raise ValueError("action must be a non-empty string of at most 64 characters")
        if target == "robot_arm" and action not in {"point_to_object", "stop"}:
            raise ValueError("simulation permits robot_arm actions: point_to_object, stop")
        if action == "point_to_object" and not isinstance(parameters.get("track_id"), int):
            raise ValueError("point_to_object requires an integer track_id")

        record = {
            "target": target,
            "action": action,
            "parameters": parameters,
            "mode": "simulation",
            "timestamp": time(),
        }
        self.actions.append(record)
        return {"status": "simulated_action_completed", "action": record}
