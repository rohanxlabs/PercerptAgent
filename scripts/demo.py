"""Reproducible perception -> memory -> tools -> simulated-action demo.

The policy mode is deliberately deterministic and clearly labeled: it is an
offline fallback for demonstrating the integration without an LLM credential.
It never represents itself as LLM reasoning or physical robot control.
"""

import argparse
import json
from pathlib import Path

import cv2

from agent.memory import EpisodicMemory
from agent.tools import ToolExecutor
from perception.detector import YOLODetector
from perception.scene_state import SceneState
from perception.stream_handler import StreamHandler
from utils.visualizer import draw_detections, draw_hud


def run_policy(scene: SceneState, memory: EpisodicMemory, tools: ToolExecutor) -> dict:
    """Act only on a newly observed tracked person; all actions remain simulated."""
    for event in scene.consume_events():
        memory.add_scene_event(event)
        if event["event"] == "ENTER" and event["class"] == "person":
            result = tools.execute("send_command", {
                "target": "robot_arm", "action": "point_to_object",
                "parameters": {"track_id": event["track_id"]},
            })
            memory.add_tool_result("send_command", result)
            return {"decision": "New person detected; queued a simulated pointing action.", "tool_result": json.loads(result)}
    return {"decision": "No new actionable person event."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the verified offline PerceptAgent integration demo.")
    parser.add_argument("--source", required=True, help="Video file path or camera index")
    parser.add_argument("--frames", type=int, default=60, help="Maximum frames to process")
    parser.add_argument("--output", default="artifacts/demo.mp4", help="Annotated video path")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stream, detector, scene = StreamHandler(source=source), YOLODetector(), SceneState()
    memory, tools = EpisodicMemory(), ToolExecutor(scene)
    writer, last_result = None, None
    stream.start()
    try:
        for index, (frame, timestamp) in enumerate(stream):
            if index >= args.frames:
                break
            detected = detector.detect(frame, timestamp)
            scene.update(detected)
            last_result = run_policy(scene, memory, tools)
            annotated = draw_hud(draw_detections(frame, detected), scene.get_snapshot(), last_result)
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
            writer.write(annotated)
    finally:
        stream.stop()
        if writer:
            writer.release()

    print(json.dumps({"output": str(output), "decision": last_result, "memory": memory.get_context(), "actions": tools.actuator.actions}, indent=2))


if __name__ == "__main__":
    main()
