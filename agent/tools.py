import json
import time
from agent.action import SimulatedActuator
from utils.logger import get_logger

logger = get_logger("tools")

# ── Tool schemas (Anthropic tool_use format) ──────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "log_event",
        "description": "Log a notable scene event to the event store.",
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                "message": {"type": "string", "description": "Human-readable event description"},
                "objects_involved": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Class names of objects involved",
                },
            },
            "required": ["severity", "message"],
        },
    },
    {
        "name": "trigger_alert",
        "description": "Trigger an alert for high-priority situations requiring immediate attention.",
        "input_schema": {
            "type": "object",
            "properties": {
                "alert_type": {
                    "type": "string",
                    "enum": ["intrusion", "crowd", "object_left", "unknown_object", "custom"],
                },
                "description": {"type": "string"},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
            },
            "required": ["alert_type", "description", "confidence"],
        },
    },
    {
        "name": "query_object_history",
        "description": "Query how long a class of object has been present and its recent behavior.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "YOLO class name e.g. 'person'"},
            },
            "required": ["class_name"],
        },
    },
    {
        "name": "annotate_frame",
        "description": "Request a custom annotation to be drawn on the current frame.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "color": {"type": "string", "enum": ["red", "green", "yellow", "blue", "white"]},
                "track_id": {"type": "integer", "description": "Optional: attach to specific tracked object"},
            },
            "required": ["label", "color"],
        },
    },
    {
        "name": "send_command",
        "description": "Send a command to an external system (robot, alarm, relay, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "System to command e.g. 'robot_arm', 'alarm', 'camera_ptz'"},
                "action": {"type": "string"},
                "parameters": {"type": "object"},
            },
            "required": ["target", "action"],
        },
    },
]


# ── Tool executor ──────────────────────────────────────────────────────────────

class ToolExecutor:
    def __init__(self, scene_state=None, actuator=None):
        self.scene_state = scene_state
        self._event_log: list[dict] = []
        self._alerts: list[dict] = []
        self._annotations: list[dict] = []
        self._commands: list[dict] = []
        self.actuator = actuator or SimulatedActuator()

    def execute(self, tool_name: str, tool_input: dict) -> str:
        validation_error = self._validate(tool_name, tool_input)
        if validation_error:
            return json.dumps({"error": validation_error})
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**tool_input)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    @staticmethod
    def _validate(tool_name: str, tool_input: dict) -> str | None:
        schema = next((item["input_schema"] for item in TOOL_SCHEMAS if item["name"] == tool_name), None)
        if schema is None:
            return f"Unknown tool: {tool_name}"
        if not isinstance(tool_input, dict):
            return "Tool input must be a JSON object"
        missing = [key for key in schema.get("required", []) if key not in tool_input]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}"
        for name, value in tool_input.items():
            prop = schema["properties"].get(name)
            if prop is None:
                return f"Unexpected argument: {name}"
            expected = prop.get("type")
            if expected == "string" and not isinstance(value, str):
                return f"{name} must be a string"
            if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                return f"{name} must be a number"
            if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                return f"{name} must be an integer"
            if expected == "array" and not isinstance(value, list):
                return f"{name} must be an array"
            if expected == "object" and not isinstance(value, dict):
                return f"{name} must be an object"
            if "enum" in prop and value not in prop["enum"]:
                return f"{name} must be one of: {', '.join(prop['enum'])}"
        if tool_name == "trigger_alert" and not 0 <= tool_input["confidence"] <= 1:
            return "confidence must be between 0 and 1"
        return None

    def _tool_log_event(self, severity: str, message: str, objects_involved: list = None) -> dict:
        entry = {
            "severity": severity,
            "message": message,
            "objects": objects_involved or [],
            "timestamp": time.time(),
        }
        self._event_log.append(entry)
        logger.info(f"[{severity.upper()}] {message}")
        return {"status": "logged", "entry": entry}

    def _tool_trigger_alert(self, alert_type: str, description: str, confidence: float) -> dict:
        alert = {
            "type": alert_type,
            "description": description,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        self._alerts.append(alert)
        logger.warning(f"ALERT [{alert_type}] conf={confidence:.2f}: {description}")
        return {"status": "alert_triggered", "alert": alert}

    def _tool_query_object_history(self, class_name: str) -> dict:
        if self.scene_state is None:
            return {"error": "No scene state available"}
        history = self.scene_state.get_object_history(class_name)
        return {
            "class_name": class_name,
            "active_instances": len(history),
            "objects": history,
        }

    def _tool_annotate_frame(self, label: str, color: str, track_id: int = None) -> dict:
        annotation = {"label": label, "color": color, "track_id": track_id}
        self._annotations.append(annotation)
        return {"status": "annotation_queued", "annotation": annotation}

    def _tool_send_command(self, target: str, action: str, parameters: dict = None) -> dict:
        cmd = {"target": target, "action": action, "parameters": parameters or {}}
        self._commands.append(cmd)
        logger.info(f"COMMAND → {target}: {action} params={parameters}")
        # Hook: replace with actual robot/relay interface
        return self.actuator.execute(target, action, cmd["parameters"])
