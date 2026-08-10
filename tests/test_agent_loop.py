import pytest
import json
from unittest.mock import MagicMock, patch
from agent.tools import ToolExecutor
from agent.memory import EpisodicMemory


def test_tool_executor_log_event():
    executor = ToolExecutor()
    result = json.loads(executor.execute("log_event", {
        "severity": "warning",
        "message": "Person detected in restricted zone",
        "objects_involved": ["person"]
    }))
    assert result["status"] == "logged"
    assert len(executor._event_log) == 1


def test_tool_executor_trigger_alert():
    executor = ToolExecutor()
    result = json.loads(executor.execute("trigger_alert", {
        "alert_type": "intrusion",
        "description": "Unknown person at gate",
        "confidence": 0.88
    }))
    assert result["status"] == "alert_triggered"
    assert executor._alerts[0]["type"] == "intrusion"


def test_tool_executor_send_command():
    executor = ToolExecutor()
    result = json.loads(executor.execute("send_command", {
        "target": "robot_arm",
        "action": "point_to_object",
        "parameters": {"track_id": 3}
    }))
    assert result["status"] == "simulated_action_completed"
    assert result["action"]["mode"] == "simulation"


def test_episodic_memory():
    mem = EpisodicMemory(max_events=5)
    mem.add_decision("Triggered alert for crowding")
    mem.add_decision("Sent command to PTZ camera")
    ctx = mem.get_context()
    assert "AGENT_DECISION" in ctx
    assert "Triggered alert" in ctx


def test_memory_overflow():
    mem = EpisodicMemory(max_events=3)
    for i in range(10):
        mem.add_decision(f"Decision {i}")
    ctx = mem.get_context()
    # Only last 3 should be in memory
    assert "Decision 9" in ctx
    assert "Decision 0" not in ctx
