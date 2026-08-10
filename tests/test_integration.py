import json

from agent.memory import EpisodicMemory
from agent.tools import ToolExecutor
from perception.detector import Detection, FrameResult
from perception.scene_state import SceneState


def test_scene_memory_tool_action_flow():
    scene = SceneState()
    detection = Detection(7, 0, "person", 0.95, [1, 2, 30, 40], [15, 21, 29, 38], 1102)
    scene.update(FrameResult(1, 10.0, [detection]))

    memory = EpisodicMemory()
    for event in scene.consume_events():
        memory.add_scene_event(event)
    executor = ToolExecutor(scene)
    result = json.loads(executor.execute("send_command", {
        "target": "robot_arm", "action": "point_to_object", "parameters": {"track_id": 7},
    }))
    memory.add_tool_result("send_command", json.dumps(result))

    assert result["status"] == "simulated_action_completed"
    assert result["action"]["mode"] == "simulation"
    assert "SCENE_EVENT" in memory.get_context()
    assert "TOOL_RESULT" in memory.get_context()


def test_invalid_tool_arguments_are_rejected():
    result = json.loads(ToolExecutor().execute("trigger_alert", {
        "alert_type": "intrusion", "description": "bad confidence", "confidence": 2,
    }))
    assert "error" in result
