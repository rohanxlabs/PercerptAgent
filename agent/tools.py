import json
import time
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
    def __init__(self, scene_state=None):
        self.scene_state = scene_state
        self._event_log: list[dict] = []
        self._alerts: list[dict] = []
        self._annotations: list[dict] = []
        self._commands: list[dict] = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**tool_input)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

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
        return {"status": "command_sent", "command": cmd}